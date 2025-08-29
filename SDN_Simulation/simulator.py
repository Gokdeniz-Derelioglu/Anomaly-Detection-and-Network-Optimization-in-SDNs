# simulator.py
# Realtime network simulator with Pygame visuals.
# - Nodes (switches) and links (latency, bandwidth, congestion)
# - Packets moving along paths; link cost depends on latency & congestion
# - Multi-route topology (6–7+ distinct routes from S to D)
# - Hooks for LSTM suspicion scores and DQN actions
#
# Controls:
#   SPACE  : pause/resume
#   N      : inject a burst of flows
#   1 / 2  : decrease / increase background flow rate
#   D      : toggle "policy mode" (Shortest-Path vs. Random Among K-shortest)
#   V      : toggle per-link utilization labels
#   R      : reset sim (same topology)
#
# Requirements: pygame
#   pip install pygame

import math
import random
import time
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json


import pygame

# ------------- Config -------------
WIDTH, HEIGHT = 1200, 700
FPS = 60
FONT_NAME = "consolas"

# Link cost weights
LAT_WEIGHT = 1.0
CONGEST_WEIGHT = 0.6

# Background traffic
BASE_FLOWS_PER_SEC = 0.8
FLOW_SIZE_BYTES = (500, 2000)  # uniform range

# Visualization
NODE_RADIUS = 16
PACKET_RADIUS = 4

# ------------- Data Structures -------------

class Node:
    def __init__(self, nid, pos, kind="switch"):
        self.id = nid
        self.pos = pos  # (x,y)
        self.kind = kind  # 'switch' or 'host'

class Link:
    def __init__(self, a, b, latency_ms=5.0, bandwidth_mbps=100.0):
        self.a = a  # node id
        self.b = b  # node id
        self.latency_ms = latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        # Sim state
        self.in_flight = 0              # packets currently on link
        self.utilization = 0.0          # 0..1 visual helper
        self.last_util_update = 0.0     # time accumulator

    def other(self, nid):
        return self.b if nid == self.a else self.a

    def travel_time_sec(self, pkt_size_bytes, inflight_hint=None):
        """Approximate per-packet travel time = prop delay + serialization + congestion factor."""
        inflight = self.in_flight if inflight_hint is None else inflight_hint
        prop = self.latency_ms / 1000.0
        ser = (pkt_size_bytes * 8) / (self.bandwidth_mbps * 1_000_000.0)  # seconds
        # Congestion stretch: each in-flight adds +10% serialization time (simple model)
        stretch = 1.0 + 0.1 * max(0, inflight - 1)
        return prop + ser * stretch

class Packet:
    def __init__(self, pid, src, dst, size_bytes, path_nodes, spawn_t, suspicion=0.0):
        self.id = pid
        self.src = src
        self.dst = dst
        self.size = size_bytes
        self.path = path_nodes[:]  # list of node ids
        self.segment_index = 0     # which edge in the path we're on
        self.progress = 0.0        # 0..1 along current link
        self.spawn_t = spawn_t
        self.finish_t = None
        self.suspicion = suspicion
        self.dropped = False
        self.drop_reason = ""

    def current_nodes(self):
        if self.segment_index >= len(self.path) - 1:
            return None
        return self.path[self.segment_index], self.path[self.segment_index + 1]

class Graph:
    def __init__(self):
        self.nodes = {}              # nid -> Node
        self.links = {}              # (min,max) -> Link
        self.adj = defaultdict(list) # nid -> list[(neighbor, link_key)]

    def add_node(self, nid, pos, kind="switch"):
        self.nodes[nid] = Node(nid, pos, kind)

    def add_link(self, a, b, latency_ms=5.0, bandwidth_mbps=100.0):
        key = tuple(sorted((a, b)))
        self.links[key] = Link(a, b, latency_ms, bandwidth_mbps)
        self.adj[a].append((b, key))
        self.adj[b].append((a, key))

    def link_key(self, a, b):
        return tuple(sorted((a, b)))

    def get_link(self, a, b):
        return self.links[self.link_key(a, b)]

    def neighbors(self, nid):
        return [x for x, _ in self.adj[nid]]

    def dijkstra_costs(self, src, pkt_size):
        """Dijkstra with dynamic link costs (latency + congestion)."""
        import heapq
        dist = {nid: math.inf for nid in self.nodes}
        prev = {nid: None for nid in self.nodes}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, key in self.adj[u]:
                if key not in self.links:  # Skip removed links
                    continue
                link = self.links[key]
                # Cost = latency_weight * latency + congest_weight * (in_flight / bandwidth)
                # Normalize congestion by bandwidth scale
                c_latency = (link.latency_ms / 1000.0) * LAT_WEIGHT
                c_cong = (link.in_flight / max(1.0, link.bandwidth_mbps)) * CONGEST_WEIGHT
                w = c_latency + c_cong
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def shortest_path(self, src, dst, pkt_size):
        _, prev = self.dijkstra_costs(src, pkt_size)
        if prev[dst] is None and src != dst:
            return None
        # reconstruct
        cur = dst
        rev = [cur]
        while cur != src:
            cur = prev[cur]
            if cur is None:
                return None
            rev.append(cur)
        return list(reversed(rev))

    def k_shortest_paths(self, src, dst, k, pkt_size):
        """Yen's algorithm (simple, not optimized). Uses dynamic costs snapshot."""
        sp = self.shortest_path(src, dst, pkt_size)
        if not sp:
            return []
        A = [sp]
        import heapq
        B = []
        
        for k_i in range(1, k):
            prev_path = A[-1]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[: i + 1]
                removed_links = []  # Store (key, link_obj) pairs
                removed_adj = []    # Store (node_a, node_b, key) for restoration
                
                # Remove edges that would replicate previous A paths with same root_path
                for p in A:
                    if p[: i + 1] == root_path and i + 1 < len(p):
                        a = p[i]
                        b = p[i + 1]
                        key = self.link_key(a, b)
                        
                        # Only remove if link exists
                        if key in self.links:
                            # Store for restoration
                            removed_links.append((key, self.links[key]))
                            
                            # Remove from links dict
                            del self.links[key]
                            
                            # Remove from adjacency lists
                            self.adj[a] = [(n, k2) for (n, k2) in self.adj[a] if k2 != key]
                            self.adj[b] = [(n, k2) for (n, k2) in self.adj[b] if k2 != key]
                            
                            # Store adjacency info for restoration
                            removed_adj.append((a, b, key))
                
                # Spur path from spur_node to dst
                spur_path = self.shortest_path(spur_node, dst, pkt_size)
                if spur_path and len(spur_path) > 1:
                    candidate = root_path[:-1] + spur_path
                    # Check if this candidate is already in A or B
                    if candidate not in A and candidate not in [path for _, path in B]:
                        # cost snapshot: total weighted cost right now
                        cost = 0.0
                        valid_path = True
                        for u, v in zip(candidate[:-1], candidate[1:]):
                            link_key = self.link_key(u, v)
                            if link_key not in self.links:
                                valid_path = False
                                break
                            link = self.links[link_key]
                            c_latency = (link.latency_ms / 1000.0) * LAT_WEIGHT
                            c_cong = (link.in_flight / max(1.0, link.bandwidth_mbps)) * CONGEST_WEIGHT
                            cost += c_latency + c_cong
                        
                        if valid_path:
                            heapq.heappush(B, (cost, candidate))
                
                # Restore removed links
                for key, link_obj in removed_links:
                    self.links[key] = link_obj
                
                for a, b, key in removed_adj:
                    self.adj[a].append((b, key))
                    self.adj[b].append((a, key))
                    
            if not B:
                break
            cost, path = heapq.heappop(B)
            A.append(path)
        return A

# ------------- Topology -------------

def build_multiroute_topology():
    """
    12-node topology with ~7+ distinct S->D routes.
    S = H0, D = H1 (hosts connected to edge switches)
    Core mesh creates many alternate routes with differing lat/bw.
    """
    g = Graph()
    # Hosts
    g.add_node("H0", (120, HEIGHT // 2), kind="host")   # Source host
    g.add_node("H1", (WIDTH - 120, HEIGHT // 2), kind="host")  # Dest host

    # Edge & core switches
    coords = {
        "A": (250, 180), "B": (250, 520),
        "C": (450, 140), "D": (450, 360), "E": (450, 580),
        "F": (650, 180), "G": (650, 520),
        "H": (850, 140), "I": (850, 360), "J": (850, 580),
    }
    for nid, pos in coords.items():
        g.add_node(nid, pos, kind="switch")

    # Host links (good bandwidth, small latency)
    g.add_link("H0", "A", latency_ms=1.5, bandwidth_mbps=1000)
    g.add_link("H1", "I", latency_ms=1.5, bandwidth_mbps=1000)

    # Fabric links (mix lat/bw to create diverse routes)
    def L(a, b, lat, bw): g.add_link(a, b, latency_ms=lat, bandwidth_mbps=bw)

    # Left fan
    L("A", "C", 2, 500); L("A", "D", 3, 300); L("B", "D", 2, 400); L("B", "E", 3, 500)
    L("C", "D", 2, 400); L("D", "E", 2, 400)

    # Middle
    L("C", "F", 3, 400); L("D", "F", 2, 300); L("D", "G", 3, 300); L("E", "G", 2, 400)
    L("F", "G", 2, 500)

    # Right fan
    L("F", "H", 2, 400); L("F", "I", 3, 300); L("G", "I", 2, 300); L("G", "J", 3, 400)
    L("H", "I", 2, 400); L("I", "J", 2, 400)

    # This creates many S→…→…→D combinations; k-shortest will find 7+ easily
    return g, "H0", "H1"

# ------------- Policy / Hooks -------------


#-------------------------------------------
# Input Policy 
#-------------------------------------------
import torch
import sys, os
from real_project.finishedModel import AdvancedLSTM

# ✅ Load model ONCE globally (not inside function, so it doesn’t reload every call)
MODEL_PATH = "saved_models/lstm_sdn_model1.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = AdvancedLSTM(
    input_dim=78,     # number of features in your CICIDS input (minus label)
    hidden_dim=64,
    num_layers=2,
    output_dim=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# ---- instantiate DQNAgent (global) ----
# state_dim must match build_packet_state length. For default build_packet_state:
# state_vector = [sus, curr_idx, dst_idx] + max_neighbors*3 neighbor features
MAX_NEIGHBORS = 6
state_dim = 1 + 1 + 1 + MAX_NEIGHBORS * 3  # sus, curr_idx, dst_idx + neighbor features
action_dim = MAX_NEIGHBORS  # keep consistent with build_packet_state

DQN_MODEL_PATH = "saved_models/dqn_agent.pth"  # change path as needed
dqn_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device, model_path=None, eps=0.05)
# optionally: dqn_agent.load(DQN_MODEL_PATH) if you have one



def prepare_features(packet_features, seq_len=8):
    """
    Convert single packet (feature vector) into model-ready format.
    If model expects sequences, buffer multiple packets before passing them here.
    """
    # Convert to tensor shape [1, seq_len, input_dim]
    features_tensor = torch.tensor(packet_features, dtype=torch.float32)
    
    # If it's just one packet (1D), expand to sequence length
    if features_tensor.ndim == 1:
        features_tensor = features_tensor.unsqueeze(0).repeat(seq_len, 1)
    
    features_tensor = features_tensor.unsqueeze(0).to(device)  # [batch=1, seq_len, input_dim]
    return features_tensor


def suspicion_score_hook(packet_features) -> float:
    """
    Take packet features -> return suspicion score in [0,1]
    """
    with torch.no_grad():
        model_input = prepare_features(packet_features)
        suspicion_score = torch.sigmoid(model(model_input)).item()
    return suspicion_score

# ---------------- DQN AGENT (skeleton) ----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, device=None, model_path=None, eps=0.1):
        self.device = device or torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eps = eps  # epsilon for greedy fallback

        # Simple MLP Q-network (replace architecture as needed)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                self.net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.net.eval()
            except Exception as e:
                print("DQNAgent: failed to load model:", e)

    def q_values(self, state_tensor):
        # state_tensor: torch.tensor([state_dim]) or [batch,state_dim]
        self.net.eval()
        with torch.no_grad():
            st = state_tensor.to(self.device).float()
            if st.dim() == 1:
                st = st.unsqueeze(0)
            q = self.net(st)
        return q.cpu().numpy()

    def select_action(self, state_vector, candidate_count):
        """
        state_vector: 1D numeric vector (numpy array or list)
        candidate_count: number of neighbors / candidate actions (int)
        returns: chosen index in range(candidate_count)
        """
        import numpy as np
        # Epsilon-greedy fallback if agent untrained
        if np.random.rand() < self.eps:
            return random.randrange(candidate_count)

        st = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
        q = self.q_values(st)[0]  # shape (action_dim,)
        # if action_dim > candidate_count, take top candidate_count and choose max among them
        q = q[:candidate_count]
        return int(np.argmax(q))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()


def dqn_action_hook(state_vector, candidate_neighbors, policy_mode="SHORTEST"):
    """
    state_vector: 1D list/np.array produced by build_packet_state
    candidate_neighbors: list of neighbor node ids (e.g., ['C','D','E'])
    Returns: index into candidate_neighbors
    """
    # If simulation policy forces RANDOM_K, follow that:
    if policy_mode == "RANDOM_K":
        return random.randrange(len(candidate_neighbors))

    # Ensure state_vector length matches dqn_agent.expected dim by padding zeros if needed
    sv = list(state_vector)
    if len(sv) < dqn_agent.state_dim:
        sv = sv + [0.0] * (dqn_agent.state_dim - len(sv))

    # Let DQNAgent pick an index; it returns an index within action_dim
    chosen_idx = dqn_agent.select_action(sv, candidate_count=len(candidate_neighbors))
    chosen_idx = max(0, min(chosen_idx, len(candidate_neighbors) - 1))
    return chosen_idx


# ------------- Simulator -------------

class Simulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SDN Network Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, 16)

        self.graph, self.host_src, self.host_dst = build_multiroute_topology()
        self.packets = []
        self.next_pid = 1

        self.running = True
        self.paused = False
        self.show_util = True
        self.policy_mode = "SHORTEST"  # or "RANDOM_K"

        self.flows_per_sec = BASE_FLOWS_PER_SEC
        self.flow_accum = 0.0

        # thresholds (from your three-zone cell)
        self.lower_thresh = 0.25
        self.upper_thresh = 0.75

        self.stats = {
            "delivered": 0,
            "dropped": 0,
            "quarantined": 0,
            "avg_delay": 0.0,
            "samples": 0,
        }
        
        # Recent packet events for display (keep last 10)
        self.recent_events = deque(maxlen=10)
        self.event_display_time = 3.0  # seconds to show each event

    # 3-zone decision
    def decision_from_suspicion(sus, lower_thresh, upper_thresh):
        """
        returns:
        "ROUTE"     -> suspicion < lower_thresh
        "QUARANTINE"-> lower_thresh <= suspicion < upper_thresh
        "DROP"      -> suspicion >= upper_thresh
        """
        if sus >= upper_thresh:
            return "DROP"
        elif sus >= lower_thresh:
            return "QUARANTINE"
        else:
            return "ROUTE"

    def choose_quarantine_target(graph, src):
        # Example: pick the host_dst as quarantine or add a dedicated quarantine node in topology.
        # If you have a quarantined sink node id (e.g., "QUARANTINE"), prefer that.
        if "QUARANTINE" in graph.nodes:
            return "QUARANTINE"
        # fallback: route to destination host (so it queues there rather than being forwarded)
        # (you can change this logic to route to an admin honeypot)
        return None


    # ---- traffic ----
    def maybe_spawn_background(self, dt):
        self.flow_accum += dt * self.flows_per_sec
        while self.flow_accum >= 1.0:
            self.flow_accum -= 1.0
            self.inject_flow()

    # ---- inject_flow ----
    def inject_flow(self, src=None, dst=None, size=None):
        src = src or self.host_src
        dst = dst or self.host_dst
        size = size or random.randint(*FLOW_SIZE_BYTES)

        # Prepare minimal feature vector for LSTM (you may want to craft this better)
        features = {
            "size": size,
            "src": src,
            "dst": dst,
            "timestamp": time.time(),
        }
        sus = suspicion_score_hook(features)  # returns float in [0,1]

        decision = decision_from_suspicion(sus, self.lower_thresh, self.upper_thresh)

        if decision == "DROP":
            self.stats["dropped"] += 1
            self.recent_events.append({
                "type": "DROPPED",
                "packet_id": self.next_pid,
                "suspicion": sus,
                "reason": f"High suspicion ({sus:.3f} >= {self.upper_thresh})",
                "timestamp": time.time(),
                "size": size
            })
            self.next_pid += 1
            return

        if decision == "QUARANTINE":
            self.stats["quarantined"] += 1
            self.recent_events.append({
                "type": "QUARANTINED",
                "packet_id": self.next_pid,
                "suspicion": sus,
                "reason": f"Medium suspicion ({sus:.3f})",
                "timestamp": time.time(),
                "size": size
            })
            # You still forward the packet, but mark it as quarantined; the DQN may choose quarantine-specific path.
            # (we keep packet in simulation but flagged)
            pkt = Packet(self.next_pid, src, dst, size, path_nodes=[src], spawn_t=time.time(), suspicion=sus)
            pkt.quarantined = True
            self.next_pid += 1
            self.packets.append(pkt)
            return

        # else decision == "ROUTE" -> normal routing with DQN for path choice
        pkt = Packet(self.next_pid, src, dst, size, path_nodes=[src], spawn_t=time.time(), suspicion=sus)
        self.next_pid += 1
        self.packets.append(pkt)


    def build_packet_state(pkt, graph, max_neighbors=6):
        """
        Build a state vector for a DQN per packet at its current node.

        Features included:
        - Suspicion score
        - Current node one-hot (or ID normalized)
        - Destination node one-hot (or ID normalized)
        - Neighbor links: for up to max_neighbors neighbors:
            * latency (ms)
            * bandwidth (Mbps)
            * congestion (in-flight / bandwidth)
        - If fewer than max_neighbors, pad with zeros
        """

        current_node = pkt.path[pkt.segment_index] if pkt.segment_index < len(pkt.path) else pkt.path[-1]
        dst_node = pkt.dst

        # --- Node encoding (simple ID -> normalized float) ---
        node_ids = list(graph.nodes.keys())
        node_idx_map = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_ids)

        curr_idx = node_idx_map[current_node] / max(1, num_nodes-1)
        dst_idx = node_idx_map[dst_node] / max(1, num_nodes-1)

        # --- Suspicion score ---
        sus = pkt.suspicion

        # --- Neighbor links ---
        neighbors = graph.neighbors(current_node)
        neighbor_features = []
        for n in neighbors[:max_neighbors]:
            link = graph.get_link(current_node, n)
            latency = link.latency_ms / 1000.0      # seconds
            bw = link.bandwidth_mbps / 1000.0      # scale down to ~1
            cong = link.in_flight / max(1.0, link.bandwidth_mbps)
            neighbor_features.extend([latency, bw, cong])

        # Pad if fewer than max_neighbors
        while len(neighbor_features) < max_neighbors*3:
            neighbor_features.extend([0.0, 0.0, 0.0])

        # --- Combine all into 1D vector ---
        state_vector = [sus, curr_idx, dst_idx] + neighbor_features
        return state_vector


    # ---- step_packets ----
    def step_packets(self, dt):
        done = []
        for pkt in self.packets:
            seg = pkt.current_nodes()

            # If first step (only src in path), select next hop
            if seg is None:
                current_node_id = pkt.path[-1]
                neighbors = self.graph.neighbors(current_node_id)
                if not neighbors:
                    # dead-end - drop
                    pkt.dropped = True
                    pkt.drop_reason = "No neighbors at spawn"
                    done.append(pkt)
                    continue

                # Build DQN state
                state_vector = self.build_packet_state(pkt, self.graph) if hasattr(self, 'build_packet_state') else build_packet_state(pkt, self.graph)
                # choose action via DQN hook (pass simulator policy_mode for consistency)
                chosen_neighbor_idx = dqn_action_hook(state_vector, neighbors, policy_mode=self.policy_mode)
                chosen_neighbor_idx = max(0, min(chosen_neighbor_idx, len(neighbors) - 1))
                next_hop = neighbors[chosen_neighbor_idx]

                # Append to path and start first link
                pkt.path.append(next_hop)
                link = self.graph.get_link(current_node_id, next_hop)
                link.in_flight += 1
                pkt.progress = 0.0
                continue

            # Regular progress along current link
            a, b = seg
            link = self.graph.get_link(a, b)
            travel = link.travel_time_sec(pkt.size)
            progress_rate = dt / travel if travel > 0 else 1.0
            pkt.progress += progress_rate

            if pkt.progress >= 1.0:
                # leave link
                link.in_flight = max(0, link.in_flight - 1)
                pkt.segment_index += 1
                pkt.progress = 0.0

                # reached destination?
                if pkt.path[-1] == pkt.dst:
                    pkt.finish_t = time.time()
                    done.append(pkt)
                else:
                    # choose next hop dynamically
                    current_node_id = pkt.path[pkt.segment_index]
                    neighbors = [n for n in self.graph.neighbors(current_node_id) if n not in pkt.path]
                    if not neighbors:
                        # dead-end, drop
                        pkt.dropped = True
                        pkt.drop_reason = "No valid next hop"
                        done.append(pkt)
                        continue
                    state_vector = build_packet_state(pkt, self.graph)
                    chosen_neighbor_idx = dqn_action_hook(state_vector, neighbors)
                    chosen_neighbor_idx = max(0, min(chosen_neighbor_idx, len(neighbors) - 1))
                    next_hop = neighbors[chosen_neighbor_idx]
                    pkt.path.append(next_hop)
                    self.graph.get_link(current_node_id, next_hop).in_flight += 1

        # finalize delivered/dropped
        for pkt in done:
            self.packets.remove(pkt)
            if pkt.dropped:
                self.stats["dropped"] += 1
            else:
                self.stats["delivered"] += 1
                delay = pkt.finish_t - pkt.spawn_t
                self.stats["samples"] += 1
                n = self.stats["samples"]
                self.stats["avg_delay"] += (delay - self.stats["avg_delay"]) / n

    def decay_utilization(self, dt):
        # Update a simple utilization metric for link coloring.
        for key, link in self.graph.links.items():
            # target utilization ~ clamp(in_flight/10)
            target = min(1.0, link.in_flight / 10.0)
            link.utilization += (target - link.utilization) * min(1.0, dt * 3.0)

    # ---- drawing ----
    def draw(self):
        self.screen.fill((18, 18, 24))

        # Draw links
        for key, link in self.graph.links.items():
            n1 = self.graph.nodes[link.a]
            n2 = self.graph.nodes[link.b]
            color = self.util_color(link.utilization)
            pygame.draw.line(self.screen, color, n1.pos, n2.pos, max(2, int(2 + 6 * link.utilization)))

            if self.show_util:
                midx = (n1.pos[0] + n2.pos[0]) // 2
                midy = (n1.pos[1] + n2.pos[1]) // 2
                txt = f"{link.latency_ms:.0f}ms | {link.bandwidth_mbps:.0f}Mb | q={link.in_flight}"
                self.draw_text(txt, (midx, midy - 14), (180, 180, 200), center=True)

        # Draw nodes
        for nid, node in self.graph.nodes.items():
            x, y = node.pos
            if node.kind == "host":
                color = (90, 200, 120) if nid == self.host_src else (200, 90, 120)
                pygame.draw.rect(self.screen, color, (x - NODE_RADIUS, y - NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2), border_radius=6)
            else:
                color = (80, 140, 220)
                pygame.draw.circle(self.screen, color, node.pos, NODE_RADIUS)
            self.draw_text(nid, (x, y + NODE_RADIUS + 12), (210, 210, 220), center=True)

        # Draw packets
        for pkt in self.packets:
            seg = pkt.current_nodes()
            if not seg:
                continue
            a = self.graph.nodes[seg[0]].pos
            b = self.graph.nodes[seg[1]].pos
            px = a[0] + (b[0] - a[0]) * pkt.progress
            py = a[1] + (b[1] - a[1]) * pkt.progress
            # color by suspicion
            c = self.susp_color(pkt.suspicion)
            pygame.draw.circle(self.screen, c, (int(px), int(py)), PACKET_RADIUS)

        # HUD
        hud_lines = [
            f"[SPACE] pause: {self.paused}",
            f"[1/2] flow rate: {self.flows_per_sec:.2f}/s",
            f"[D] policy: {self.policy_mode}",
            f"[V] link labels: {self.show_util}",
            f"[N] inject burst   [R] reset",
            f"Delivered: {self.stats['delivered']}  Dropped: {self.stats['dropped']}  Quarantined: {self.stats['quarantined']}  Avg delay: {self.stats['avg_delay']:.3f}s",
            f"Thresholds  lower={self.lower_thresh:.2f}  upper={self.upper_thresh:.2f}",
            f"Packets in flight: {len(self.packets)}",
        ]
        for i, line in enumerate(hud_lines):
            self.draw_text(line, (12, 12 + i * 18), (230, 230, 240))

        # Draw recent packet events at bottom of screen
        self.draw_packet_events()

        pygame.display.flip()
    
    def draw_packet_events(self):
        """Display recent dropped/quarantined packets at bottom of screen"""
        current_time = time.time()
        # Filter events that are still within display time
        active_events = [
            event for event in self.recent_events 
            if current_time - event["timestamp"] < self.event_display_time
        ]
        
        if not active_events:
            return
            
        # Draw background panel
        panel_height = len(active_events) * 20 + 20
        panel_rect = pygame.Rect(10, HEIGHT - panel_height - 10, WIDTH - 20, panel_height)
        pygame.draw.rect(self.screen, (40, 40, 50, 200), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 120), panel_rect, 2)
        
        # Draw events
        y_offset = HEIGHT - panel_height - 5
        for i, event in enumerate(reversed(active_events[-5:])):  # Show last 5 events
            color = (255, 100, 100) if event["type"] == "DROPPED" else (255, 200, 100)
            age = current_time - event["timestamp"]
            alpha = max(100, 255 - int(age / self.event_display_time * 155))  # Fade out over time
            
            event_text = f"PKT#{event['packet_id']} {event['type']}: {event['reason']} ({event['size']}B)"
            
            # Create surface with alpha for fading effect
            text_surf = self.font.render(event_text, True, (*color, alpha))
            self.screen.blit(text_surf, (20, y_offset + i * 20))

    def draw_text(self, text, pos, color, center=False):
        surf = self.font.render(text, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surf, rect)

    @staticmethod
    def util_color(u):
        # green -> yellow -> red
        u = max(0.0, min(1.0, u))
        r = int(60 + 180 * u)
        g = int(200 - 140 * u)
        b = 80
        return (r, g, b)

    @staticmethod
    def susp_color(s):
        # suspicion 0..1: blue->purple->red
        s = max(0.0, min(1.0, s))
        r = int(80 + 160 * s)
        g = int(120 - 90 * s)
        b = int(220 - 120 * s)
        return (r, g, b)

    # ---- main loop ----
    def run(self):
        last = time.time()
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            now = time.time()
            elapsed = now - last
            last = now

            self.handle_events()
            if not self.paused:
                self.maybe_spawn_background(elapsed)
                self.step_packets(elapsed)
                self.decay_utilization(elapsed)

            self.draw()

        pygame.quit()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_1:
                    self.flows_per_sec = max(0.0, self.flows_per_sec - 0.2)
                elif e.key == pygame.K_2:
                    self.flows_per_sec += 0.2
                elif e.key == pygame.K_n:
                    for _ in range(10):
                        self.inject_flow()
                elif e.key == pygame.K_d:
                    self.policy_mode = "RANDOM_K" if self.policy_mode == "SHORTEST" else "SHORTEST"
                elif e.key == pygame.K_v:
                    self.show_util = not self.show_util
                elif e.key == pygame.K_r:
                    self.reset()

    def reset(self):
        self.graph, self.host_src, self.host_dst = build_multiroute_topology()
        self.packets.clear()
        self.next_pid = 1
        self.stats = {"delivered": 0, "dropped": 0, "quarantined": 0, "avg_delay": 0.0, "samples": 0}
        self.flow_accum = 0.0
        self.recent_events.clear()


import pandas as pd, time

class CSVTrafficStream:
    def __init__(self, csv_path, feature_columns=None, delay=0.0, loop=False):
        self.df = pd.read_csv(csv_path)
        self.cols = feature_columns or list(self.df.columns)
        self.index = 0
        self.delay = delay
        self.loop = loop

    def next_packet(self):
        if self.index >= len(self.df):
            if self.loop:
                self.index = 0
            else:
                return None
        row = self.df.iloc[self.index]
        self.index += 1
        if self.delay:
            time.sleep(self.delay)
        # map row to injector args (size, src, dst...) — you will need to adjust this
        size = int(row.get("size", random.randint(*FLOW_SIZE_BYTES)))
        # optionally get src/dst mapping if included in CSV
        return {"size": size, "src": row.get("src", None), "dst": row.get("dst", None), "raw_row": row.to_dict()}


# ------------- Entry -------------

if __name__ == "__main__":
    Simulator().run()