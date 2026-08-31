#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <numeric>
#include <functional>
#include <iomanip>
#include <cstdlib>
#include <chrono>

using namespace std;

// ---------------------------------------------------------------------------
// Vodak et al. (2018) pheromone bounds. Their parameter tables (Section 3) give
// p_min, p_max = 0, 1 for both test networks. The shipped baseline
// (2018modifiedACO_opt.cpp) instead uses a 0.01 floor; this build restores the
// published values so the literal Eq (12) marking can be exercised.
// ---------------------------------------------------------------------------
static const double PMIN = 0.0;
static const double PMAX = 1.0;

// ==========================================
// 1. UTILITY FUNCTIONS
// Helper functions for common, repetitive tasks
// ==========================================

// This function removes leading and trailing spaces (or tabs/newlines) from a string.
// It is useful when reading data from files to avoid formatting issues.
string trim(const string &s) {
    size_t start = s.find_first_not_of(" \t\r\n"); // Find the first non-space character
    if (start == string::npos) return "";          // If the string is all spaces, return empty
    size_t end = s.find_last_not_of(" \t\r\n");    // Find the last non-space character
    return s.substr(start, end - start + 1);       // Return the cleaned-up substring
}

// This function splits a single string into a list of strings based on a specific character (delimiter).
// Example: splitting "A,B,C" by ',' gives ["A", "B", "C"].
vector<string> split(const string &s, char delim) {
    vector<string> tokens;         // List to hold our split words
    istringstream stream(s);       // Treat the string like an input stream
    string token;
    // Read from the stream until the delimiter is hit
    while (getline(stream, token, delim)) {
        tokens.push_back(trim(token)); // Clean up spaces and add to our list
    }
    return tokens;
}

// This function splits a string based on whitespace (spaces or tabs).
// It is specifically used for reading datasets where spaces act as dividers.
vector<string> split_whitespace(const string &s) {
    vector<string> tokens;
    istringstream stream(s);
    string token;
    // The >> operator automatically skips random amounts of spaces
    while (stream >> token) {
        tokens.push_back(token); // Add each word to our list
    }
    return tokens;
}

// Return the environment variable `name` if set and non-empty, else `fallback`.
// Lets one binary point at either network / config without recompiling.
static string env_or(const char *name, const string &fallback) {
    const char *v = getenv(name);
    return (v && *v) ? string(v) : fallback;
}

// Master RNG seed. ACO_SEED makes a run reproducible; unset, the old
// random_device behaviour is kept so nothing that ran before changes now.
// Resolved once and cached so every consumer sees the same value.
static unsigned int master_seed() {
    static const unsigned int s = []() -> unsigned int {
        const char *v = getenv("ACO_SEED");
        if (v && *v) {
            try { return static_cast<unsigned int>(stoul(v)); } catch (...) {}
        }
        return random_device{}();
    }();
    return s;
}

// ==========================================
// 2. DATA STRUCTURE DEFINITIONS
// Memory blueprints for holding our parsed data
// ==========================================



// Holds the information for a single street/link connecting two nodes
struct LinkRow {
    int init_node, term_node;                    // The start node and end node of the link
    double capacity, length, free_flow_time;     // Link properties. Free flow time is our main "cost"
    // BPR volume-delay parameters (defaults -> factor 1.0 when data is absent)
    double b = 0.0, power = 1.0, voc = 0.0;
};

// Holds information about a path that has been damaged/blocked
struct BlockedLink {
    int u, v;                       // The two nodes connected by this blocked street
    double travel_time, repair_time;// How long it takes to naturally travel it vs how long to repair it
};

// Format strictly used for reading the user's disaster CSV file
struct BlockedInput {
    int nodeA, nodeB;   // Nodes connected by damaged street
    double repairTime;  // How long it takes to fix it
};


// ==========================================
// 3. GRAPH ENGINE (THE MAP)
// Represents the entire road network and handles pathfinding
// ==========================================
class Graph {
public:
    // This is our main blueprint. It maps a Node ID to a list of (Neighbor ID, Travel Time).
    // Basically: "From Node 1, I can go to Node 2 (takes 5 mins) or Node 4 (takes 10 mins)"
    map<int, vector<pair<int, double>>> adj;

    // ---- OPTIMIZATION CACHES (pure memoisation; results identical to the
    // original). The graph is built once then only READ during the ACO, so we
    // cache the expensive lookups and invalidate them on any structural change.
    //   sp_cache : one full single-source Dijkstra result per source node, so the
    //              same source is never recomputed (get_full_path calls it for
    //              every hop of every unit of every candidate solution).
    //   ew_cache : direct edge weight, so get_edge_weight is O(1) not a scan.
    struct SPResult { map<int, double> dist; map<int, vector<int>> paths; };
    mutable map<int, SPResult> sp_cache;
    mutable map<pair<int, int>, double> ew_cache;

    void invalidate_caches() { sp_cache.clear(); ew_cache.clear(); }

    // Adds a two-way (undirected) road between node u and node v
    void add_edge(int u, int v, double weight) {
        adj[u].push_back({v, weight}); // Add v as a destination from u
        adj[v].push_back({u, weight}); // Add u as a destination from v
        invalidate_caches();           // structure changed -> caches are stale
    }

    // Searches our map to find how long it takes to travel directly between u and v
    double get_edge_weight(int u, int v) const {
        auto ck = ew_cache.find({u, v});
        if (ck != ew_cache.end()) return ck->second;
        double w = 1.0; // Default fallback if not found
        auto it = adj.find(u); // Look up our starting node
        if (it != adj.end()) {
            // Check all connected neighbors
            for (auto &p : it->second)
                // If the neighbor matches our destination 'v', return the weight (time)
                if (p.first == v) { w = p.second; break; }
        }
        ew_cache[{u, v}] = w;
        return w;
    }

    // Completely removes a road between u and v. Used when simulating the disaster breaking roads.
    void remove_edge(int u, int v) {
        // A helper tool to delete a specific destination from a node's list
        auto remove_from = [](vector<pair<int, double>> &vec, int target) {
            vec.erase(remove_if(vec.begin(), vec.end(), [target](const pair<int, double> &p) {
                return p.first == target;
            }), vec.end());
        };
        // Remove the road from both ends to ensure it's fully broken
        if (adj.count(u)) remove_from(adj[u], v);
        if (adj.count(v)) remove_from(adj[v], u);
        invalidate_caches();           // structure changed -> caches are stale
    }

    // Returns a list of every single street existing in the current map
    vector<pair<int, int>> edges() const {
        vector<pair<int, int>> result;
        for (auto &kv : adj)              // Go through every node
            for (auto &p : kv.second)     // Go through all its destinations
                result.push_back({kv.first, p.first});
        return result;
    }

    // Returns a unique list consisting of every node ID that exists in this map
    set<int> nodes_set() const {
        set<int> ns;
        for (auto &kv : adj) ns.insert(kv.first);
        return ns;
    }

    // Creates an identical standalone copy of the entire map. Useful so we don't accidentally edit the original.
    Graph copy() const {
        Graph g;
        g.adj = adj;
        return g;
    }

    // DIJKSTRA'S ALGORITHM (cached core)
    // Computes the shortest path from `source` to EVERY node exactly once and
    // memoises the result. Identical algorithm, tie-breaks and paths as the
    // original; callers that hit the same source (which the ACO does millions of
    // times) get an O(1) reference back instead of a fresh search.
    const SPResult &cached_sssp(int source) const {
        auto found = sp_cache.find(source);
        if (found != sp_cache.end()) return found->second;

        SPResult res;
        map<int, double> &distances = res.dist;
        map<int, vector<int>> &paths = res.paths;
        using PII = pair<double, int>; // Pair holding (Total Time So Far, Current Node)
        // A "Priority Queue" works like a to-do list that continuously puts the fastest/shortest paths at the top.
        priority_queue<PII, vector<PII>, greater<PII>> pq;

        distances[source] = 0.0;     // Time to get to the start node from the start node is 0
        paths[source] = {source};    // The path to the start node is just the start node itself
        pq.push({0.0, source});      // Add the starting point to our to-do list

        // Loop until there are no more nodes left to check
        while (!pq.empty()) {
            auto [d, u] = pq.top(); // Get the node with the current shortest travel time
            pq.pop();               // Remove it from the to-do list

            // If we've already found a faster way to get here, skip this calculation
            if (d > distances[u]) continue;

            auto it = adj.find(u);
            if (it == adj.end()) continue; // Skip if this node has nowhere to go

            // Look at every road leaving our current node
            for (auto &[v, w] : it->second) {
                double nd = d + w; // Calculate total time to get to neighbor 'v' (Time to 'u' + road time 'w')

                // If we haven't visited 'v' yet, OR if this new route is faster than the old route:
                if (!distances.count(v) || nd < distances[v]) {
                    distances[v] = nd;           // Save the new fastest time
                    paths[v] = paths[u];         // Copy the path it took to get here
                    paths[v].push_back(v);       // Add 'v' to the end of that path
                    pq.push({nd, v});            // Add 'v' to the to-do list so we can check its neighbors later
                }
            }
        }
        auto ins = sp_cache.emplace(source, move(res));
        return ins.first->second;
    }

    // Backwards-compatible wrapper (used by the one-time setup callers). Copies the
    // cached result out into the caller's maps, exactly as the original returned.
    void single_source_dijkstra(int source, map<int, double> &distances, map<int, vector<int>> &paths) const {
        const SPResult &r = cached_sssp(source);
        distances = r.dist;
        paths = r.paths;
    }

    // CONNECTED COMPONENTS
    // Checks the map to figure out which nodes can actually reach each other.
    // When roads break, the map might shatter into separate "islands" (components) that can't reach one another.
    vector<set<int>> connected_components() const {
        set<int> all_nodes = nodes_set();
        set<int> visited;                // Tracker to remember who we've already categorized
        vector<set<int>> components;     // Will hold the final lists of islands

        // Loop over every node on the map
        for (int node : all_nodes) {
            if (visited.count(node)) continue; // Skip if it's already part of an island
            
            set<int> comp;      // Create a fresh island list
            queue<int> q;       // Standard line (Queue) for processing neighbors
            q.push(node);
            visited.insert(node);

            // Expanding the island by grabbing all neighbors of neighbors 
            while (!q.empty()) {
                int cur = q.front(); q.pop(); // Take the guy at the front of the line
                comp.insert(cur);             // Add him to the island

                auto it = adj.find(cur);
                if (it != adj.end()) {
                    // Look at all his direct neighbors
                    for (auto &[nb, w] : it->second) {
                        // If the neighbor isn't already claimed, claim him for the island and add him to the line
                        if (!visited.count(nb)) {
                            visited.insert(nb);
                            q.push(nb);
                        }
                    }
                }
            }
            components.push_back(comp); // Save this finalized island and look for any remaining nodes
        }
        return components;
    }
};


// ==========================================
// 4. DATA LOADING MODULE
// Translates flat files (CSVs) into our C++ structures
// ==========================================

// Loads the standard map information (connecting roads)
void load_sioux_falls_data(const string &net_file, vector<LinkRow> &links_out) {
    
    // --- Load the Links (Streets) ---
    ifstream fnet(net_file); // Open net file
    if (!fnet.is_open()) { cerr << "Error: " << net_file << endl; exit(1); } // Safety check
    string line;
    getline(fnet, line); // Read the header string (e.g. init_node,term_node,capacity...)
    
    // Split the header by comma so we know which column is which
    vector<string> headers = split(line, ',');
    int init_idx = -1, term_idx = -1, cap_idx = -1, len_idx = -1, fft_idx = -1;
    int b_idx = -1, power_idx = -1, voc_idx = -1;
    
    // Loop through headers to figure out their exact positions in the file.
    // Robust to: UTF-8 BOM on the first header, lower/upper case, spaces vs
    // underscores ("init node" == "init_node"), and parenthetical units
    // ("length (meters)" == "length", "free flow time (minute)" == "free_flow_time").
    for (size_t i = 0; i < headers.size(); ++i) {
        string h = trim(headers[i]);
        // Strip a leading UTF-8 BOM (EF BB BF) if present
        if (h.size() >= 3 && (unsigned char)h[0] == 0xEF && (unsigned char)h[1] == 0xBB && (unsigned char)h[2] == 0xBF)
            h = h.substr(3);
        transform(h.begin(), h.end(), h.begin(), ::tolower); // case-insensitive
        // Drop any parenthetical unit, e.g. "length (meters)" -> "length"
        size_t paren = h.find('(');
        if (paren != string::npos) h = trim(h.substr(0, paren));
        // Normalise spaces to underscores so "init node" matches "init_node"
        replace(h.begin(), h.end(), ' ', '_');

        if      (h == "init_node" || h == "init") init_idx = i;
        else if (h == "term_node" || h == "term") term_idx = i;
        else if (h == "capacity")  cap_idx = i;
        else if (h == "length")    len_idx = i;
        else if (h == "free_flow_time" || h == "free_flow") fft_idx = i;
        else if (h == "b")         b_idx = i;
        else if (h == "power")     power_idx = i;
        else if (h == "volume/capacity") voc_idx = i;
    }

    // Hard fail early with a clear message if the essential columns are missing,
    // instead of silently loading 0 links.
    if (init_idx == -1 || term_idx == -1) {
        cerr << "ERROR: could not find init/term node columns in " << net_file
             << ". Header seen: "; for (auto &h : headers) cerr << "[" << h << "] "; cerr << "\n";
        exit(1);
    }

    // Read the actual data rows
    while (getline(fnet, line)) {
        line = trim(line);
        if (line.empty()) continue;
        auto tokens = split(line, ','); // Break apart by comma since it's a CSV

        // Ensure this row has enough columns to avoid crashing
        if ((int)tokens.size() > max({init_idx, term_idx, cap_idx, len_idx, fft_idx})) {
            try {
                LinkRow lr;
                // Map the parsed strings back to our LinkRow struct based on the header indices we found
                lr.init_node = stoi(tokens[init_idx]);
                lr.term_node = stoi(tokens[term_idx]);
                lr.capacity = (cap_idx != -1 && !tokens[cap_idx].empty()) ? stod(tokens[cap_idx]) : 0;
                lr.length = (len_idx != -1 && !tokens[len_idx].empty()) ? stod(tokens[len_idx]) : 0;
                lr.free_flow_time = (fft_idx != -1 && !tokens[fft_idx].empty()) ? stod(tokens[fft_idx]) : 1.0;
                // BPR volume-delay parameters (absent -> defaults keep the factor at 1.0)
                lr.b = (b_idx != -1 && b_idx < (int)tokens.size() && !tokens[b_idx].empty()) ? stod(tokens[b_idx]) : 0.0;
                lr.power = (power_idx != -1 && power_idx < (int)tokens.size() && !tokens[power_idx].empty()) ? stod(tokens[power_idx]) : 1.0;
                lr.voc = (voc_idx != -1 && voc_idx < (int)tokens.size() && !tokens[voc_idx].empty()) ? stod(tokens[voc_idx]) : 0.0;
                links_out.push_back(lr); // Add it to our final list
            } catch (...) { /* skip malformed row */ }
        }
    }

    // Print a quick summary to ensure we imported correctly
    cout << "\n--- Verifying Loaded Network (Links) Data ---" << endl;
    cout << "Columns found: ";
    for (auto &h : headers) cout << h << " "; cout << "\n";
    cout << "Total Links: " << links_out.size() << endl;
    
    // Print just a preview of the first 5 records
    for(size_t i=0; i<min((size_t)5, links_out.size()); ++i) {
        auto &lr = links_out[i];
        cout << lr.init_node << "\t" << lr.term_node << "\t" << lr.capacity << "\t" << lr.length << "\t" << lr.free_flow_time << "\n";
    }
    cout << "..." << endl;
    cout << "-------------------------------------\n" << endl;
}

// Loads the user's custom disaster file showing which streets are broken
vector<BlockedInput> load_disaster_input(const string &csv_file) {
    ifstream fin(csv_file);
    if (!fin.is_open()) { cerr << "Error: " << csv_file << " not found." << endl; exit(1); }
    vector<BlockedInput> result;
    string line;
    getline(fin, line); // Skip header logic assumes it exists
    
    while (getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        auto tokens = split(line, ',');
        // We expect NodeA, NodeB, and Repair Time
        if (tokens.size() >= 3)
            result.push_back({stoi(tokens[0]), stoi(tokens[1]), stod(tokens[2])});
    }
    return result;
}

// ==========================================
// 5. DISASTER SCENARIO ENGINE
// Physically breaks the roads in our map and identifies the resulting islands
// ==========================================
void apply_disaster_scenario(const vector<LinkRow> &links, const vector<BlockedInput> &blocked_input,
                             Graph &G_full, vector<BlockedLink> &actual_blocked_links, map<int, int> &node_comp_map) {
                                 
    // Record original travel times for every street
    map<pair<int, int>, double> link_travel_times;
    
    // Fill the master Intact Graph (G_full) with all standard links
    for (auto &row : links) {
        double cost = row.free_flow_time;
        if (cost == 0) cost = row.length; // Fallback in case free flow time is missing

        // BPR congestion adjustment: travel = t0 * (1 + b * (V/C)^power).
        // voc==0 -> factor 1.0, so links without V/C data stay at free-flow.
        cost *= (1.0 + row.b * pow(row.voc, row.power));

        // Ensure pairs are always ordered logically (smallest Node ID first). Prevents duplicate logic bugs.
        pair<int, int> edge_key = {min(row.init_node, row.term_node), max(row.init_node, row.term_node)};
        if (!link_travel_times.count(edge_key)) link_travel_times[edge_key] = cost;
        
        // Add road to our map
        G_full.add_edge(row.init_node, row.term_node, cost);
    }

    // Process the repair demands given by the disaster file
    set<pair<int, int>> user_blocks;              // Set of all broken roads
    map<pair<int, int>, double> block_info;       // Map of broken road -> time required to fix it
    
    for (auto &bi : blocked_input) {
        pair<int, int> key = {min(bi.nodeA, bi.nodeB), max(bi.nodeA, bi.nodeB)};
        user_blocks.insert(key); // Flag this route as broken
        
        // If a road shows up multiple times in the disaster file, keep the maximum repair time
        if (block_info.count(key)) block_info[key] = max(block_info[key], bi.repairTime);
        else block_info[key] = bi.repairTime;
    }

    Graph G_broken = G_full.copy(); // Copy the healthy map to start breaking it
    actual_blocked_links.clear();
    set<pair<int, int>> processed;  // Keep track to avoid deleting a road twice

    // Iterate through all existing roads on the healthy map
    for (auto &[u, v] : G_full.edges()) {
        pair<int, int> key = {min(u, v), max(u, v)};
        
        // If this road is flagged as broken, and we haven't broken it yet...
        if (user_blocks.count(key) && !processed.count(key)) {
            processed.insert(key);
            G_broken.remove_edge(u, v); // Delete the road
            
            // Get original drive time so the repair crews know what it feels like to drive it once fixed
            double t_travel = link_travel_times.count(key) ? link_travel_times[key] : 1.0;
            // Mark it thoroughly for the algorithm
            actual_blocked_links.push_back({key.first, key.second, t_travel, block_info[key]});
        }
    }

    // Now that the graph is broken, figure out the resulting isolated groups of nodes
    auto components = G_broken.connected_components();
    node_comp_map.clear();
    
    // Assign every single node a unique 'Island ID'
    for (size_t i = 0; i < components.size(); i++) {
        for (int node : components[i]) node_comp_map[node] = i;
    }

    // Output the results so the user can verify the disaster happened successfully
    cout << "Disaster Scenario Applied: Graph broken into " << components.size() << " components.\n";
    for (size_t i = 0; i < components.size(); i++) {
        vector<int> sorted_comp(components[i].begin(), components[i].end());
        sort(sorted_comp.begin(), sorted_comp.end());
        cout << "Component " << i << ": [";
        for (size_t j = 0; j < sorted_comp.size(); j++) {
            if (j > 0) cout << ", ";
            cout << sorted_comp[j];
        }
        cout << "]\n";
    }
}

// ==========================================
// 6. GRAPH REDUCTION
// Simplifies a massive city map down into a smaller map where:
// Nodes = Points that connect different islands (bridges)
// Edges = The combined driving/repairing paths between those points
// ==========================================
void build_reduced_graph(const Graph &G_full, const vector<BlockedLink> &blocked_links, const map<int, int> &node_comp_map,
                         int base,
                         vector<int> &boundary_nodes_out, map<pair<int, int>, pair<double, double>> &dist_matrix_out,
                         map<int, pair<double, double>> &base_dists_out) {

    // STEP 1: Find Boundary Nodes (Points on the edge of an island)
    set<int> boundary_set;
    for (auto &link : blocked_links) {
        // If both nodes connected by this shattered road belong to known islands
        if (node_comp_map.count(link.u) && node_comp_map.count(link.v)) {
            boundary_set.insert(link.u);
            boundary_set.insert(link.v);
        }
    }
    boundary_nodes_out.assign(boundary_set.begin(), boundary_set.end());
    
    cout << "Set of Boundary Nodes: ";
    for (int n : boundary_nodes_out) cout << n << " ";
    cout << "\nBoundary nodes identified: " << boundary_nodes_out.size() << endl;

    // Fast lookup tool to ask "If I traverse {node A, node B}, what is the repair penalty?"
    map<pair<int, int>, double> blocked_lookup;
    for (auto &x : blocked_links) blocked_lookup[{min(x.u, x.v), max(x.u, x.v)}] = x.repair_time;

    dist_matrix_out.clear(); // Clear all old routes

    // STEP 2: Calculate the best paths jumping between boundary nodes
    for (int start_node : boundary_nodes_out) {
        map<int, double> lengths;    // Holds total driving time to get somewhere
        map<int, vector<int>> paths; // Holds the literal breadcrumb trail of node IDs
        
        // Find shortest paths leaving 'start_node' based ONLY on standard driving speeds
        G_full.single_source_dijkstra(start_node, lengths, paths);

        // Verification prints mirroring the python dict prints for the user
        // cout << "\n--- Results for Start Node: " << start_node << " ---" << endl;
        // cout << "DISTANCES: {";
        // int c=0;
        // for(auto& kv : lengths) { if(c++>0)cout<<", "; cout << kv.first << ": " << kv.second; }
        // cout << "}\nPATHS: {";
        // c=0;
        // for(auto& kv : paths) { 
        //     if(c++>0)cout<<", "; 
        //     cout << kv.first << ": [";
        //     for(size_t j=0; j<kv.second.size(); j++) { if(j>0)cout<<", "; cout << kv.second[j]<<".0"; }
        //     cout << "]";
        // }
        // cout << "}\n";

        // Now, evaluate the result for every possible OTHER boundary node destination
        for (int end_node : boundary_nodes_out) {
            if (start_node == end_node) continue; // Can't travel to yourself
            if (!paths.count(end_node)) continue; // Can't go there if unreachable

            auto &path = paths[end_node]; // The sequence of nodes from Start to End
            
            double total_ts = 0.0; // TS = Travel Sum (Total standard driving time)
            double total_tr = 0.0; // TR = Time for Repair (Total time repairing required on this path)
            
            // Loop through the actual street-by-street path
            for (size_t i = 0; i < path.size() - 1; i++) {
                int u = path[i], v = path[i + 1];
                
                // Driving cost
                total_ts += G_full.get_edge_weight(u, v);
                
                // Repair cost (If this street is actually in our broken list, add the penalty)
                pair<int, int> edge_key = {min(u, v), max(u, v)};
                if (blocked_lookup.count(edge_key)) total_tr += blocked_lookup[edge_key];
            }
            
            // We ONLY record this path in our final reduced matrix if it involves repairing something.
            // Otherwise, we're just driving randomly without fixing things.
            if (total_tr > 0) dist_matrix_out[{start_node, end_node}] = {total_ts, total_tr};
        }
    }
    cout << "Reduced graph edges created: " << dist_matrix_out.size() << endl;

    // STEP 3: Base station as an ARTIFICIAL COMPONENT (2018 paper).
    // Compute repair-aware distances from the base to every boundary node:
    //   .first  = travel time, .second = repair time of broken edges crossed.
    // Broken edges INCIDENT to the base are treated as artificial zero-repair
    // depot edges (repair 0). This is kept in a SEPARATE base->boundary map
    // (not merged into dist_matrix), so the base is never a selectable
    // destination and no cheap boundary->base edge competes with boundary->boundary.
    base_dists_out.clear();
    {
        map<int, double> lengths;
        map<int, vector<int>> paths;
        G_full.single_source_dijkstra(base, lengths, paths);
        for (int bn : boundary_nodes_out) {
            if (!paths.count(bn)) {
                base_dists_out[bn] = {numeric_limits<double>::infinity(), 0.0};
                continue;
            }
            auto &path = paths[bn];
            double ts = 0.0, tr = 0.0;
            for (size_t i = 0; i + 1 < path.size(); i++) {
                int u = path[i], v = path[i + 1];
                ts += G_full.get_edge_weight(u, v);
                pair<int, int> edge_key = {min(u, v), max(u, v)};
                // base-incident broken edge => artificial depot edge, repair 0
                if (blocked_lookup.count(edge_key) && u != base && v != base)
                    tr += blocked_lookup[edge_key];
            }
            base_dists_out[bn] = {ts, tr};
        }
    }
}

// ==========================================
// 7. ROUTE VALIDATION HELPERS
// Functions mapping Ant routes back to reality to ensure costs are accurate
// ==========================================

// An Ant route might look like: Base -> B_Node1 -> B_Node2 -> Base
// This helper expands that back into: Base -> Node7 -> Node8 -> B_Node1 -> Node 5 ...
vector<int> get_full_path(const Graph &G_full, const vector<int> &route) {
    if(route.empty()) return {};
    
    // Start our full path off with the first node from the high-level route
    vector<int> full_path = {route[0]};
    
    // Iterate over the "hops" (e.g. B_Node1 to B_Node2)
    for (size_t k = 0; k < route.size() - 1; k++) {
        int u = route[k], v = route[k + 1];
        if(u==v) continue; // Unnecessary hop

        // HOT PATH: read the memoised Dijkstra result for source u by reference
        // (no per-hop search, no whole-map copy). Same path the original produced.
        const Graph::SPResult &r = G_full.cached_sssp(u);
        auto pit = r.paths.find(v);
        if (pit != r.paths.end()) {
            const auto &p = pit->second; // Retrieve the specific path between these two points
            // Paste it into our main track, skipping the first point so we don't have duplicates
            full_path.insert(full_path.end(), p.begin() + 1, p.end());
        } else {
            // Panic fallback (should technically never hit this if Dijkstra succeeded previously)
            full_path.push_back(v);
        }
    }
    return full_path;
}

// Exactly mirrors the newly added compute_path_TS_TR_with_repair python update
// This is critical to ensure repair crews aren't "spending time" fixing the exact same road twice!
pair<double, double> compute_path_TS_TR_with_repair(const vector<int> &path, const Graph &G, const map<pair<int, int>, double> &blocked_lookup, set<pair<int, int>> &repaired_edges) {
    double total_ts = 0.0, total_tr = 0.0;
    
    // Evaluate every single hop in the physical path
    for (size_t i = 0; i < path.size() - 1; i++) {
        int u = path[i], v = path[i + 1];
        
        // Add driving time
        total_ts += G.get_edge_weight(u, v);
        
        pair<int, int> edge_key = {min(u, v), max(u, v)};
        
        // If this road is broken AND we haven't already repaired it during this route:
        if (blocked_lookup.count(edge_key) && !repaired_edges.count(edge_key)) {
            total_tr += blocked_lookup.at(edge_key); // Incur the repair cost penalty
            repaired_edges.insert(edge_key);         // Mark it as permanently fixed so we don't pay for it again
        }
    }
    return {total_ts, total_tr}; // Return the final values for logging
}

// Injects the {"-"} visual string marker immediately before any road segment that is broken
vector<string> mark_repaired_edges(const vector<int> &path, const map<pair<int, int>, double> &blocked_lookup) {
    if(path.empty()) return {};
    vector<string> marked = {to_string(path[0])};
    for (size_t i = 0; i < path.size() - 1; i++) {
        int u = path[i], v = path[i+1];
        pair<int, int> edge_key = {min(u, v), max(u, v)};
        
        // If this exact jump is broken, insert our symbol into the output log
        if (blocked_lookup.count(edge_key)) marked.push_back("{-}");
        marked.push_back(to_string(v));
    }
    return marked;
}

// Per-unit cost + repair attribution for the whole fleet.
struct UnitEval {
    vector<int> full_path;                  // expanded physical driving route
    double travel = 0.0;                    // free-flow travel over every traversal
    double repair = 0.0;                    // repair of edges THIS unit actually repaired
    vector<pair<int, int>> repaired;        // those edges, in the order this unit reached them
};

// FLEET EVALUATOR (single source of truth for both the optimiser and the report).
// Each broken edge is repaired exactly once across the whole fleet, attributed
// CHRONOLOGICALLY: the unit that physically reaches the edge first (lowest cumulative
// travel time to its near endpoint; ties broken by unit index) repairs it; every
// other unit just drives across it for free. Edges with repair 0 (e.g. the base's
// artificial depot edges) are skipped — they are not "repairs".
vector<UnitEval> evaluate_fleet(const vector<vector<int>> &routes, const Graph &G,
                                const map<pair<int, int>, double> &blocked_lookup) {
    int m = (int)routes.size();
    vector<UnitEval> out(m);

    // `key` = normalised {min,max} for de-dup/lookup; `dir` = the actual (from,to)
    // direction this unit drove across the edge, which is what we display.
    struct Event { double t; int unit; pair<int, int> key; pair<int, int> dir; };
    vector<Event> events;

    for (int i = 0; i < m; i++) {
        out[i].full_path = get_full_path(G, routes[i]);
        const auto &fp = out[i].full_path;
        double cum = 0.0;                       // cumulative travel time along this unit's route
        set<pair<int, int>> seen;               // first crossing of each broken edge by this unit
        for (size_t k = 0; k + 1 < fp.size(); k++) {
            int u = fp[k], v = fp[k + 1];
            pair<int, int> ek = {min(u, v), max(u, v)};
            // Register an arrival event at the moment the unit REACHES the broken edge
            // (before traversing it), once per unit, only for edges that cost repair.
            if (blocked_lookup.count(ek) && blocked_lookup.at(ek) > 0.0 && !seen.count(ek)) {
                events.push_back({cum, i, ek, {u, v}}); // keep the driven direction (u -> v)
                seen.insert(ek);
            }
            cum += G.get_edge_weight(u, v);
            out[i].travel += G.get_edge_weight(u, v);
        }
    }

    // Earliest arrival wins the repair; deterministic tie-break by unit index.
    sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        if (a.t != b.t) return a.t < b.t;
        return a.unit < b.unit;
    });

    set<pair<int, int>> repaired;
    for (auto &e : events) {
        if (repaired.count(e.key)) continue;    // already repaired by an earlier-arriving unit
        repaired.insert(e.key);
        out[e.unit].repaired.push_back(e.dir);  // show the actual traversal direction
        out[e.unit].repair += blocked_lookup.at(e.key);
    }
    return out;
}


// ==========================================
// 8. ANT COLONY OPTIMIZATION (ACO)
// The primary algorithm mimicking ants dropping pheromones to find the fastest global routes
// ==========================================
class ModifiedACO {
public:
    vector<int> nodes;                                      // Boundary nodes to explore
    map<pair<int, int>, pair<double, double>> dist_matrix;  // Costs memory bank
    map<int, int> comp_map;                                 // Knowing who belongs to what island
    
    int m, scenario; // Number of crews, Scenario 1 or 2
    Graph G_full;
    
    // Weights determining if ants prioritize smell (pheromones) vs logic (heuristics)
    double alpha = 1.0, beta = 1.0, rho = 0.2; 
    
    map<pair<int, int>, double> heuristics, pheromones;

    int base, artificial_base_comp_id = -1; // Base station variables
    map<int, pair<double, double>> base_dists;       // repair-aware base->boundary {travel, repair}
    map<pair<int, int>, double> blocked_lookup;      // broken edge -> repair time (for true-cost eval)

    // Keep trackers of the absolute best solutions discovered so far
    vector<vector<int>> best_routes;
    vector<double> best_times;
    vector<double> best_times_sorted;
    
    mt19937 rng; // The random number generator for ant probability rolls

    // ---- DENSE READ MIRRORS (raw-node-id indexed; node ids are small, <= ~71).
    // The maps above stay the single source of truth and all UPDATE logic
    // (evaporate/deposit/clamp, savings, init) still works on them unchanged.
    // These matrices mirror the same values for the read-hot inner loops
    // (_select_next, construct_solution move costs), turning ~600M map-of-pair
    // lookups into O(1) array reads. Values are identical to the maps.
    int NID = 0;
    int n_real_comps = 0;                          // number of distinct components (once)
    vector<vector<double>> ph_d, he_d, ts_d, tr_d; // pheromones, heuristics, dist travel, dist repair
    vector<vector<char>>   ex_d;                   // dist_matrix existence flag
    vector<int>            comp_d;                  // comp_map, dense
    vector<double>         bts_d, btr_d;            // base_dists travel / repair, dense

    // Copy the (changing) pheromone map into its dense mirror. Cheap (~few k entries).
    void sync_ph() {
        for (auto &kv : pheromones) ph_d[kv.first.first][kv.first.second] = kv.second;
    }

    // Build all dense mirrors once from the finished maps (end of construction).
    void build_dense() {
        NID = 0;
        for (auto &kv : comp_map) NID = max(NID, kv.first);
        NID += 1;
        comp_d.assign(NID, -1);
        { set<int> s; for (auto &kv : comp_map) s.insert(kv.second); n_real_comps = (int)s.size(); }
        for (auto &kv : comp_map) comp_d[kv.first] = kv.second;
        he_d.assign(NID, vector<double>(NID, 0.0));
        ph_d.assign(NID, vector<double>(NID, 0.0));
        ts_d.assign(NID, vector<double>(NID, 0.0));
        tr_d.assign(NID, vector<double>(NID, 0.0));
        ex_d.assign(NID, vector<char>(NID, 0));
        for (auto &kv : dist_matrix) {
            ex_d[kv.first.first][kv.first.second] = 1;
            ts_d[kv.first.first][kv.first.second] = kv.second.first;
            tr_d[kv.first.first][kv.first.second] = kv.second.second;
        }
        for (auto &kv : heuristics)
            he_d[kv.first.first][kv.first.second] = kv.second;
        bts_d.assign(NID, numeric_limits<double>::infinity());
        btr_d.assign(NID, 0.0);
        for (auto &kv : base_dists) { bts_d[kv.first] = kv.second.first; btr_d[kv.first] = kv.second.second; }
        sync_ph();
    }

    // Initialization block
    ModifiedACO(const vector<int> &b_nodes, const map<pair<int, int>, pair<double, double>> &d_mat, const map<int, int> &c_map,
                int num_u, int b_node, const Graph &g_f,
                const map<int, pair<double, double>> &b_dists, const map<pair<int, int>, double> &b_lookup,
                int scen = 1)
        : nodes(b_nodes), dist_matrix(d_mat), comp_map(c_map), m(num_u), scenario(scen), G_full(g_f), base(b_node),
          base_dists(b_dists), blocked_lookup(b_lookup) {

        // Kickstart our random algorithm. Seeded from ACO_SEED when set, so a
        // run can be reproduced; see master_seed(). NOTE: main() constructs this
        // class twice, once per reported scenario, and both get the same seed.
        rng.seed(master_seed());

        if (!comp_map.count(base)) { cerr << "Base missing" << endl; exit(1); }

        // The base sits in its own component, which is reached for free (the crews
        // start there). Treat it as the artificial component so ants never waste a
        // trip "visiting" it.
        artificial_base_comp_id = comp_map.at(base);

        // base_dists (repair-aware base->boundary {travel, repair}) is supplied by
        // build_reduced_graph; no travel-only recomputation here.

        // Populate initial 'logic' appeal (Heuristics). Ants logically prefer shorter paths.
        for (int i : nodes) {
            for (int j : nodes) {
                if (i != j && dist_matrix.count({i, j}))
                    // Invert the total distance. Bigger distance = Smaller appeal. Add 0.001 to prevent dividing by zero crashing it.
                    heuristics[{i, j}] = 1.0 / (dist_matrix[{i, j}].first + 0.001);
            }
            // Same logic for returning to the base station (travel component of base distance)
            heuristics[{base, i}] = 1.0 / (base_dists[i].first + 0.001);
        }

        // Clarke-Wright pre-solve shortcut. Generates an initial "decent" route organically to give ants a headstart
        auto [init_routes, init_times] = savings_init();
        
        best_routes = init_routes;
        best_times = init_times;
        best_times_sorted = init_times;
        sort(best_times_sorted.begin(), best_times_sorted.end(), greater<double>()); // Sort descending

        // Eq (11)-(12): the initial marking value, from the savings route's length.
        double lengthpath = 0; for(double t : init_times) lengthpath += t;
        double tau0 = max(PMIN, min(PMAX, 1.0 / (lengthpath + 1e-6)));

        // Vodak Sec. 2.3.2 lays the initial marking on the SAVINGS ROUTE only and
        // does not state a value for the remaining edges. Leaving them at p_min = 0
        // would make them permanently unselectable, so they start a fixed factor
        // BELOW the marking value instead. Deriving that value from tau0 rather
        // than fixing it at a constant keeps the ratio scale-free: the savings
        // route leads by 10x on any network, in any cost units.
        double off_route = tau0 / 10.0;   // savings route starts 10x above everything else
        for (int i : nodes) {
            for (int j : nodes) if (i != j) pheromones[{i, j}] = off_route;
            pheromones[{base, i}] = off_route;
        }
        for (auto &r : init_routes) {
            for (size_t i = 0; i + 1 < r.size(); i++) {
                if (pheromones.count({r[i], r[i+1]})) pheromones[{r[i], r[i+1]}] = tau0;
                if (pheromones.count({r[i+1], r[i]})) pheromones[{r[i+1], r[i]}] = tau0;
            }
        }

        // All maps are final now; build the dense read mirrors used by the hot loops.
        build_dense();
    }

    // TRUE-COST EVALUATOR (objective used by the optimiser, identical to the report).
    // Expands every route to its full physical path and costs it sharing ONE global
    // repaired_edges set across all units (index order), so each broken edge is
    // repaired exactly once across the fleet and out-and-back crossings are not
    // double-counted. Returns each unit's objective time:
    //   scenario 1 -> travel only;  scenario 2 -> travel + repair.
    vector<double> eval_route_times(const vector<vector<int>> &routes) {
        // Use the shared fleet evaluator so the objective matches the report exactly,
        // including the chronological "first to arrive repairs it" attribution.
        auto fleet = evaluate_fleet(routes, G_full, blocked_lookup);
        vector<double> times;
        times.reserve(fleet.size());
        for (const auto &uc : fleet)
            times.push_back(uc.travel + (scenario == 2 ? uc.repair : 0.0));
        return times;
    }

    // SAVINGS ALGORITHM (Clarke Wright)
    // Concept: What if everyone just drives out and back individually?
    // Then, look for pairs of tasks where merging them into one long trip "saves" the highest amount of useless driving back and forth.
    pair<vector<vector<int>>, vector<double>> savings_init() {
        
        map<int, vector<int>> routes;
        map<int, int> node_route_map;
        
        // Base case: Everyone gets a custom route consisting of Base -> Location -> Base.
        // The base node itself is the DEPOT, not a task: per the paper it forms its own
        // artificial component and is never "visited". Excluding it here (mirroring the
        // component-level savings in acoBmTSP_savings_opt.cpp) keeps every route well-formed
        // and prevents the zero-value base savings from erasing/dereferencing the depot stub.
        for (int n : nodes) { if (n == base) continue; routes[n] = {base, n, base}; node_route_map[n] = n; }

        struct Saving { double val; int i, j; };
        vector<Saving> savings;

        // Formula calculation loop to check all possible pairings (base excluded: it is the depot)
        for (int i : nodes) {
            if (i == base) continue;
            for (int j : nodes) {
                if (j == base) continue;
                if (i != j && dist_matrix.count({i, j})) {
                    // Formula Value = Dist(Base to Task A) + Dist(Base to Task B) - Dist(Task A direct to Task B)
                    savings.push_back({base_dists[i].first + base_dists[j].first - dist_matrix[{i, j}].first, i, j});
                }
            }
        }
        
        // Sort highest savings to the top of the list so we act on them first
        sort(savings.begin(), savings.end(), [](const Saving &a, const Saving &b) { return a.val > b.val; });

        // Loop through the best savings and merge routes if theoretically possible
        for (auto &sav : savings) {
            // Guard with .count() before operator[] so an already-merged (erased) route id
            // can never silently fabricate an empty vector (mirrors acoBmTSP_savings_opt.cpp).
            if (!node_route_map.count(sav.i) || !node_route_map.count(sav.j)) continue;
            int ri_id = node_route_map[sav.i], rj_id = node_route_map[sav.j];
            if (ri_id == rj_id) continue; // Skip if they're already part of the same route
            if (!routes.count(ri_id) || !routes.count(rj_id)) continue;

            auto &ri = routes[ri_id], &rj = routes[rj_id];

            // Only merge if Node I is immediately before returning to base, and Node J just left the base.
            // i.e., Route 1: [0, 5, i, 0] + Route 2: [0, j, 8, 0] -> Merged Output: [0, 5, i, j, 8, 0]
            if (ri.size() >= 2 && rj.size() >= 2 && ri[ri.size() - 2] == sav.i && rj[1] == sav.j) {
                vector<int> merged(ri.begin(), ri.end() - 1);         // Copy Route 1 but drop the terminating '0'
                merged.insert(merged.end(), rj.begin() + 1, rj.end());// Paste Route 2 but drop the starting '0'
                routes[ri_id] = merged;                               // Update the main route
                routes.erase(rj_id);                                  // Erase the old disconnected route
                
                // Remap ownership
                for (int n : merged) if (n != base) node_route_map[n] = ri_id;
            }
            
            // If the amount of resulting routes shrinks to match our crew amount constraints, stop merging completely
            if ((int)routes.size() <= m) break;
        }

        // Copy everything out into standard arrays
        vector<vector<int>> final_routes;
        for (auto &kv : routes) final_routes.push_back(kv.second);

        // Emergency loop: If we somehow still have more assignments than we have crews available, force merge remaining jobs manually.
        while ((int)final_routes.size() > m) {
            auto r1 = final_routes.back(); final_routes.pop_back();
            auto r2 = final_routes.back(); final_routes.pop_back();
            vector<int> merged(r1.begin(), r1.end() - 1);
            merged.insert(merged.end(), r2.begin() + 1, r2.end());
            final_routes.push_back(merged);
        }

        // Calculate final expected operational times using the TRUE-cost evaluator
        // (full physical path + repair-once), so the savings baseline is on the same
        // objective the ACO and the report use.
        vector<double> route_times = eval_route_times(final_routes);

        // Print initialization logs purely mirroring the Python prints
        cout << "Savings initialization produced " << final_routes.size() << " routes.\nInitial routes from Savings:\n";
        for (auto &r : final_routes) {
            cout << "["; for(size_t k=0;k<r.size();k++) cout << (k>0?", ":"") << r[k]; cout << "]\n";
        }
        cout << "route times: [";
        for (size_t k = 0; k < route_times.size(); k++) {
            if(k>0) cout<<", ";
            // Floor output: If it's x.00 perfectly, print as int, otherwise print 2 decimals
            if(floor(route_times[k]) == route_times[k]) cout << (int)route_times[k];
            else cout << round(route_times[k]*100)/100.0;
        }
        cout << "]\n----------------------------------------\n";
        return {final_routes, route_times};
    }

    // INTERNAL ANT LOGIC: Given where an ant currently is standing, choose its next destination
    int _select_next(int curr, const set<int> &visited_comps) {
        vector<int> cands; vector<double> wts; // Arrays holding 'Where we can go' vs 'Probability score of going there'

        // pow(x,1)==x, so with alpha==beta==1 (the config used throughout) skip the
        // two pow() calls per candidate — they dominate this loop, which runs
        // ~m x nodes times per component per solution. Value is identical.
        const bool pow_skip = (alpha == 1.0 && beta == 1.0);

        const vector<double> &ph_row = ph_d[curr];
        const vector<double> &he_row = he_d[curr];
        const vector<char>   &ex_row = ex_d[curr];
        for (int n : nodes) {
            // Can't go to yourself, and can't go to an island we've already cleared
            if (n == curr || visited_comps.count(comp_d[n])) continue;

            // Reachability is decided by the reduced graph and the heuristic, NOT by
            // the pheromone level. With p_min = 0 a trail can sit at zero, and such
            // an edge has to stay reachable or the ant would be locked out of it for
            // the whole run. Weight 0 candidates are kept; the sumw == 0 branch below
            // falls back to a uniform draw, exactly as the original did.
            bool reachable = ((curr == base) || ex_row[n]) && he_row[n] > 0.0;
            if (!reachable) continue;

            double p = ph_row[n], h = he_row[n];

            // Core Ant Formula calculation combining smell vs logical appeal
            double w = pow_skip ? (p * h) : (pow(p, alpha) * pow(h, beta));
            cands.push_back(n); wts.push_back(w);
        }
        
        if (cands.empty()) return -1; // -1 represents 'No valid targets' (Python None equivalent)
        
        double sumw = 0; for(double w : wts) sumw += w;
        
        // If absolutely no paths stand out, make a fully randomized guess from available options
        if (sumw == 0) return cands[uniform_int_distribution<>(0, cands.size() - 1)(rng)];
        
        // Roulette Wheel selection. Options with higher weights simply hold wider coverage in the wheel 
        discrete_distribution<> dist(wts.begin(), wts.end());
        return cands[dist(rng)];
    }

    // Generates completely new routes by sending one single 'wave' of ants through the map
    pair<vector<vector<int>>, vector<double>> construct_solution() {
        vector<int> pos(m, base);                 // Every ant starts at the base
        vector<double> times(m, 0);               // Every ant starts with a 0 cost
        vector<vector<int>> routes(m, {base});    // Keep track of routes built sequentially
        
        // The base's OWN component is reached for free (crews start there) — pre-mark
        // it visited so ants only travel to OTHER isolated components.
        set<int> visited_comps = {artificial_base_comp_id}; // = comp_map[base]

        // Keep simulating ant steps sequentially until every broken island group has been visited
        while ((int)visited_comps.size() < n_real_comps) {
            struct Move { int ant, nxt; double ts, tr, cval; };
            vector<Move> moves;

            // Ask every ant: "Hey, what move would you LIKE to make right now?"
            for (int i = 0; i < m; i++) {
                int nxt = _select_next(pos[i], visited_comps);
                if (nxt == -1) continue;

                double ts = (pos[i] == base) ? bts_d[nxt] : ts_d[pos[i]][nxt];
                double tr = (pos[i] == base) ? btr_d[nxt] : tr_d[pos[i]][nxt];
                
                // Calculate how attractive this specific decision is relative to the total cost.
                double cval = 1.0 / (times[i] + ts + (scenario == 2 ? tr : 0) + 0.001);
                moves.push_back({i, nxt, ts, tr, cval});
            }
            
            if (moves.empty()) break; // Terminate early if ants physically get stranded
            
            // Roll the dice to decide WHICH ONE SPECIFIC ANT actually gets to deploy their proposed move simultaneously
            vector<double> probs; for(auto &mv : moves) probs.push_back(mv.cval);
            auto &mv = moves[discrete_distribution<>(probs.begin(), probs.end())(rng)];
            
            // Update tracking based on the move executed
            pos[mv.ant] = mv.nxt;
            routes[mv.ant].push_back(mv.nxt);
            times[mv.ant] += mv.ts + (scenario == 2 ? mv.tr : 0);
            visited_comps.insert(comp_d[mv.nxt]);
        }

        // At the end of the shift, universally instruct every ant crew to return to the base station.
        // (The incremental `times` above only guided the ant-picket ordering.)
        for (int i = 0; i < m; i++) {
            if (pos[i] != base) times[i] += bts_d[pos[i]];
            routes[i].push_back(base);
        }

        // OBJECTIVE: replace the guidance estimate with the TRUE cost of the finished
        // routes (full physical path + repair-once across all units), so what the ACO
        // ranks/optimises equals what the report prints.
        times = eval_route_times(routes);
        return {routes, times};
    }

    // Main execution looping method executing the repeated colony runs
    pair<vector<vector<int>>, vector<double>> run(int iterations = 1000, int num_of_colonies = 30) {
        
        // Start out with out 'Savings Algorithm' values as the current best global baseline to beat
        auto cb_routes = best_routes; 
        auto cb_times = best_times; 
        auto cb_t_s = best_times_sorted;
        
        for (int iter = 0; iter < iterations; iter++) {

            // Refresh the dense pheromone mirror from the map updated at the end of
            // the previous iteration, so this iteration's ants read current values.
            sync_ph();

            // Fire off a massive cluster of ants independently to generate solutions
            std::vector<std::pair<std::vector<std::vector<int>>, std::vector<double>>> sols(num_of_colonies);
            for (int c = 0; c < num_of_colonies; c++) sols[c] = construct_solution();
            
            // Identify the single #1 best performing run in this particular colony batch
            auto sort_t = [](vector<double> t) { sort(t.begin(), t.end(), greater<double>()); return t; };
            int best_idx = 0;
            vector<double> b_key = sort_t(sols[0].second);
            
            // Loop through times looking for lexicographical minimum times (e.g. evaluating max end time first)
            for(int c = 1; c < num_of_colonies; c++) {
                auto key = sort_t(sols[c].second);
                if(key < b_key) { b_key = key; best_idx = c; }
            }
            
            // If the best run from this batch is better than the overall global history, update the history.
            if (b_key < cb_t_s) { cb_t_s = b_key; cb_routes = sols[best_idx].first; cb_times = sols[best_idx].second; }

            // Pheromone Evaporation -> Trails slowly weaken over time to prevent permanent lock-in assumptions
            for (auto &kv : pheromones) kv.second *= (1 - rho);
            
            // Reward Step - Choose via coin flip whether to encourage the "Historical Global Best" path or "Current Iteration Local Best" path
            auto &s_routes = (uniform_real_distribution<>(0, 1)(rng) < 0.5) ? cb_routes : sols[best_idx].first;
            auto &s_times = (uniform_real_distribution<>(0, 1)(rng) < 0.5) ? cb_times : sols[best_idx].second;
            
            double L = 0; for(double t : s_times) L += t;  // Eq (11): length_path
            double pmark = max(PMIN, min(PMAX, 1.0 / (L + 1e-6))); // Eq (12): marking value

            // Vodak Eq (12) is an ASSIGNMENT, not an increment: the trail on every
            // link of the chosen route is SET to the marking value.
            for (auto &r : s_routes) {
                for(size_t i=0; i<r.size()-1; i++) {
                    if (r[i] != base && r[i+1] != base) { // Never deposit on returning home trails to prevent bias locks
                        if (pheromones.count({r[i], r[i+1]})) pheromones[{r[i], r[i+1]}] = pmark;
                        if (pheromones.count({r[i+1], r[i]})) pheromones[{r[i+1], r[i]}] = pmark;
                    }
                }
            }
            
            // MMAS Pheromone Clamping - Force boundary limitations ensuring numbers never mathematically break or sink too low
            for (auto &kv : pheromones) kv.second = max(PMIN, min(PMAX, kv.second));
        }
        return {cb_routes, cb_times};
    }
};

// Utilities specifically for printing nice looking vector outputs to the command line matching terminal python outputs.
void print_vec(const vector<int> &v) {
    cout << "["; for(size_t i=0;i<v.size();i++) cout << (i>0?", ":"") << v[i]; cout << "]";
}

// Special string array printer that catches the injected marker "{-}"
void print_vec_s(const vector<string> &v) {
    cout << "["; 
    for(size_t i=0;i<v.size();i++) {
        if(i>0)cout<<", ";
        if(v[i]=="-") cout<<"'-'"; 
        else if(v[i]=="{-}") cout<<"'{-}'";
        else cout<<v[i];
    } 
    cout << "]";
}

// ==========================================
// 9b. CONFIG FILE LOADING
// (same key=value contract as the other ACO builds, so the 2018 baseline is
//  driven by config.txt / config_Sioux.txt exactly like them — no keyboard input)
// ==========================================
struct Config {
    int base_station_node_id;
    int number_of_repair_units;
    double alpha;
    double beta;
    int number_of_iterations;
};

Config load_config(const string &file) {
    ifstream f(file);
    if (!f.is_open()) {
        cerr << "ERROR: Cannot open config file: " << file << "\n"
             << "       Make sure '" << file << "' exists in the working directory.\n";
        exit(1);
    }

    map<string, string> kv;
    string line;
    int line_no = 0;
    while (getline(f, line)) {
        line_no++;
        string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue; // comment / blank

        size_t eq = t.find('=');
        if (eq == string::npos) {
            cerr << "ERROR: Malformed line " << line_no << " in " << file
                 << " (expected key=value): \"" << t << "\"\n";
            exit(1);
        }
        string key = trim(t.substr(0, eq));
        string val = trim(t.substr(eq + 1));
        if (key.empty() || val.empty()) {
            cerr << "ERROR: Missing key/value on line " << line_no << " in " << file << "\n";
            exit(1);
        }
        kv[key] = val;
    }

    vector<string> required = {
        "base_station_node_id", "number_of_repair_units",
        "alpha", "beta", "number_of_iterations"};
    for (auto &k : required)
        if (!kv.count(k)) {
            cerr << "ERROR: Missing required config key \"" << k << "\" in " << file << "\n";
            exit(1);
        }

    auto parse_int = [&](const string &key) -> int {
        try { size_t pos; int v = stoi(kv[key], &pos);
              if (pos != kv[key].size()) throw invalid_argument("trailing"); return v; }
        catch (...) { cerr << "ERROR: Config key \"" << key << "\" must be an integer, got \""
                           << kv[key] << "\" in " << file << "\n"; exit(1); }
        return 0;
    };
    auto parse_double = [&](const string &key) -> double {
        try { size_t pos; double v = stod(kv[key], &pos);
              if (pos != kv[key].size()) throw invalid_argument("trailing"); return v; }
        catch (...) { cerr << "ERROR: Config key \"" << key << "\" must be a number, got \""
                           << kv[key] << "\" in " << file << "\n"; exit(1); }
        return 0.0;
    };

    Config cfg;
    cfg.base_station_node_id = parse_int("base_station_node_id");
    cfg.number_of_repair_units = parse_int("number_of_repair_units");
    cfg.alpha = parse_double("alpha");
    cfg.beta = parse_double("beta");
    cfg.number_of_iterations = parse_int("number_of_iterations");

    if (cfg.number_of_repair_units < 1) {
        cerr << "ERROR: number_of_repair_units must be >= 1, got "
             << cfg.number_of_repair_units << "\n"; exit(1);
    }
    if (cfg.number_of_iterations < 1) {
        cerr << "ERROR: number_of_iterations must be >= 1, got "
             << cfg.number_of_iterations << "\n"; exit(1);
    }

    cout << "Loaded config from " << file << ":\n"
         << "  base_station_node_id   = " << cfg.base_station_node_id << "\n"
         << "  number_of_repair_units = " << cfg.number_of_repair_units << "\n"
         << "  alpha                  = " << cfg.alpha << "\n"
         << "  beta                   = " << cfg.beta << "\n"
         << "  number_of_iterations   = " << cfg.number_of_iterations << "\n";
    return cfg;
}

// ==========================================
// 10. MAIN PROCESS (EXECUTION START)
// ==========================================
int main() {
    
    // File paths + config file are overridable via env vars (ACO_NET / ACO_BLOCKED
    // / ACO_CONFIG) so one binary runs both networks; defaults are the Dhaka set.
    //   Sioux: ACO_NET=SiouxFalls_net.csv ACO_BLOCKED=blocked_links_generated.csv
    //          ACO_CONFIG=config_Sioux.txt
    string NET_FILE = env_or("ACO_NET", "netDhakaDataset.csv");
    string DISASTER_FILE = env_or("ACO_BLOCKED", "blocked_links_generated_for_Dhaka_Components.csv");
    const string CONFIG_FILE = env_or("ACO_CONFIG", "config.txt");

    // Wall-clock instrumentation for the scalability question (RQ3). Preprocess
    // covers parsing, the disaster application and the graph reduction, all of
    // which run once; the optimiser is timed separately inside execute_scenario.
    using clk = chrono::steady_clock;
    auto t_start = clk::now();

    // 1. Initial Parsing
    cout << "Loading Data...\n";
    vector<LinkRow> links_df;
    load_sioux_falls_data(NET_FILE, links_df);
    vector<BlockedInput> blocked_df = load_disaster_input(DISASTER_FILE);

    // 2. Destruction Phase
    Graph G_full; vector<BlockedLink> blocked_links_info; map<int, int> comp_map;
    apply_disaster_scenario(links_df, blocked_df, G_full, blocked_links_info, comp_map);

    // 2b. Configuration — base node / repair units / iterations come from the
    // config file (no interactive input), matching the other ACO builds. The base
    // is integrated as an artificial component during graph reduction below.
    cout << "\nLoading configuration...\n";
    Config cfg = load_config(CONFIG_FILE);
    int base_node = cfg.base_station_node_id;
    if (!comp_map.count(base_node)) {
        cerr << "ERROR: base_station_node_id " << base_node << " (from " << CONFIG_FILE
             << ") is not a valid node in the network.\n";
        exit(1);
    }
    cout << " Base node " << base_node << " belongs to Component " << comp_map[base_node] << "\n";

    // 3. Compression Phase (base integrated as artificial component)
    cout << "Reducing Graph...\n";
    vector<int> boundary_nodes;
    map<pair<int, int>, pair<double, double>> dist_matrix;
    map<int, pair<double, double>> base_dists; // repair-aware base->boundary distances
    build_reduced_graph(G_full, blocked_links_info, comp_map, base_node, boundary_nodes, dist_matrix, base_dists);

    // Visual printout explicitly showing TS (Travel Time) and TR (Repair Time) parameters for debugging
    // cout << "\n--- Reduced Graph Distance Matrix ---\n";
    // for (auto &kv : dist_matrix) {
    //     cout << "Edge (" << setw(2) << kv.first.first << ", " << setw(2) << kv.first.second << ") -> "
    //          << "Travel Time (TS): " << fixed << setprecision(2) << kv.second.first << ", "
    //          << "Repair Time (TR): " << kv.second.second << "\n";
    // }
    // cout << "-------------------------------------\n\n";

    int NUM_UNITS = cfg.number_of_repair_units; // number of trucks/crews (from config)

    // Quick reference table caching all physical repair spots
    map<pair<int, int>, double> blocked_lookup;
    for (auto &x : blocked_links_info) blocked_lookup[{min(x.u, x.v), max(x.u, x.v)}] = x.repair_time;

    // Artificial-component rule (2018 paper): broken edges INCIDENT to the base are
    // the depot's own doorstep — repaired for free (repair 0). Apply it to the
    // shared lookup so the optimiser, the report, and base_dists all agree.
    for (auto &kv : blocked_lookup)
        if (kv.first.first == base_node || kv.first.second == base_node)
            kv.second = 0.0;

    auto t_pre = clk::now();
    double opt_ms_reported = 0.0;   // scenario 2 is the reported result

    // Helper macro to execute identically formatted executions for ease of readability
    auto execute_scenario = [&](int scenario) {

        cout << "\n--- Running Scenario " << scenario << " (" << (scenario==1?"No":"With") << " Repair Cost) ---\n";

        // Setup class parameters
        ModifiedACO aco(boundary_nodes, dist_matrix, comp_map, NUM_UNITS, base_node, G_full, base_dists, blocked_lookup, scenario);

        // Actually run the loops and obtain final paths mapping
        auto t_run0 = clk::now();
        auto [routes, times] = aco.run(cfg.number_of_iterations, 30);
        double run_ms = chrono::duration<double, milli>(clk::now() - t_run0).count();
        if (scenario == 2) opt_ms_reported = run_ms;
        
        cout << "\nResults Scenario " << scenario << "  (min as minute) :\n";

        // Evaluate the whole fleet once: travel + repair-once with CHRONOLOGICAL
        // attribution (the first unit to reach a broken edge repairs it). This is the
        // same evaluator the optimiser uses, so per-unit "Total Cost" == "Raw ACO Time".
        auto fleet = evaluate_fleet(routes, G_full, blocked_lookup);

        for (size_t i = 0; i < routes.size(); i++) {
            const auto &uc = fleet[i];

            cout << "Unit " << i+1 << ": Time(min) " << uc.travel << ", Repair Cost(min) " << uc.repair
                 << ", Total Cost = " << uc.travel + uc.repair << " , Route: ";
            print_vec(routes[i]);
            cout << "\n        Full Path: "; print_vec(uc.full_path);
            cout << "\n        Repairs : ";
            if (uc.repaired.empty()) cout << "(none)";
            else for (auto &e : uc.repaired) cout << "(" << e.first << "-" << e.second << ") ";
            cout << "\n";
        }
        for(size_t i=0;i<times.size();i++)
        {
            cout << "Raw ACO Time Unit "
                << i+1
                << " = "
                << times[i]
                << endl;
        }
        
        // Descending array printout. Shows ending timestamps longest job to fastest job.
        vector<double> cl_times = times; sort(cl_times.begin(), cl_times.end(), greater<double>());
        cout << "Sorted Times: ["; for(size_t i=0;i<cl_times.size();i++) cout<<(i>0?", ":"")<<cl_times[i]; cout << "]\n";
        
        // Matching an extra raw list print behavior explicitly asked for in scenario 2 implementation
        if (scenario == 2) for(double t : times) cout << t << "\n";
    };

    cout << "\nfree flow time is in minutes\n";
    cout << "Time : x -> here x is sum of all link free-flow travel times along the full path\n";
    // Sequentially trigger full system evaluations based on required parameters
    execute_scenario(1);
    execute_scenario(2);

    // Timing report, on its own [timing] lines after the results so the existing
    // log parsers are unaffected. optimise_ms is scenario 2 only, the reported
    // result; scenario 1 is the no-repair-cost pass and is not counted.
    {
        auto ms = [](clk::time_point a, clk::time_point b)
        { return chrono::duration<double, milli>(b - a).count(); };
        const double pre_ms = ms(t_start, t_pre);
        const int iters = cfg.number_of_iterations;
        cout << fixed << setprecision(3)
             << "\n[timing] seed = " << master_seed()
             << "\n[timing] boundary_nodes = " << boundary_nodes.size()
             << "\n[timing] preprocess_ms = " << pre_ms
             << "\n[timing] optimise_ms = " << opt_ms_reported
             << "\n[timing] total_ms = " << ms(t_start, clk::now())
             << "\n[timing] ms_per_iteration = "
             << (iters > 0 ? opt_ms_reported / iters : 0.0)
             << "\n";
    }

    return 0; // Terminate C++ program properly
}