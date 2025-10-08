# Routing Algorithm to Improve Women's Safety

## Street Sexual Harassment as the motivation of this Master's Thesis 

Street Sexual Harassment (SSH) is a form of gender-based violence suffered by many women. Depending on the place, data on the prevalence of this issue changes a lot: 
* In Spain, 91% of women report having suffered SSH at least once in their lives, and more than half report facing it weekly.
* In the United States, 68% report experiencing it ocasionally.
* Across the European Union, 55% of women have experienced SSH since the age of 15.

### Our proposed solution: incorporating safety inputs into routing algorithms 

Modern route algorithms, as well as urban design, are tipically developed with motor vehicles in mind. The result is applications that often overlook key pedestrian needs such as safety, accessibility and confort. Most existing routing apps calculate the shortest path between two nodes (commonly using Dijkstra’s algorithm). However, pedestrians often prefer slightly longer routes if they feel safer.

Our proposal is to calculate routes that integrate safety factors such as street length, reports of harassment and prevalence of SSH. Each edge in the network is assigned a safety cost representing the risk of experiencing harassment along the segment. 

### Key challenge: the lack of data 

Street Sexual Harassment is highly underreported, creating a significant data gap. To overcome this, we generate synthetic data to simulate SSH events. This is further explained in the sections below.

## Implementation 

### Constructing the map

We use the OSMnx Python library to download the street network of central Madrid:

``` python
bbox = (-3.718, 40.4168, -3.692, 40.4368)
G = ox.graph_from_bbox(bbox, network_type='walk', simplify=False)
```
We then clean the map by removing edges inaccessible to pedestrians, and represent it as a network:
* Edges: represent street segments.
* Nodes: represent intersections between the streets.

This is the final network we are going to use:

<br /> <img width="2256" height="2546" alt="network_clean" src="https://github.com/user-attachments/assets/d16a14bd-3151-4e53-984a-3a943cc62fc8" /> 

### Developing the Safety Map

We assign a safety score to each edge based on four inputs:
* Edge length: longer streets reduce escape opportunities.
* Active incidents: reports of SSH currently taking place.
* Historical risk: areas with simulated recurring incidents.
* Safe spots: shops or institutions that can provide assistance.

#### Edge length 

Longer segments are penalized with a non-linear function. We normalize edge length by dividing by the maximum length, and raise the ratio to the power of 1.5.

```
python
# Maximum edge length
max_length = edges['length'].max()

# Function
def length_penalty(length, max_length=max_length, exponent=1.5):
    norm = length / max_length
    return min(norm ** exponent, 1.0)

# Assign as an attribute
edges['length_penalty'] = edges['length'].apply(length_penalty)

for u, v, k, data in G.edges(keys=True, data=True):
    data['length_penalty'] = length_penalty(data.get('length', 0))
```

#### Active incidents 

We simulate 15 random SSH events in the study area. Incidents are assigned to edges with a probability proportional to their length, and then placed at a random point within the selected edge.

```
python
sampled = edges_for_incidents.sample(
    n=15,
    replace=True,
    weights=edges_for_incidents['length']
)
points = sampled.geometry.apply(
    lambda line: line.interpolate(random.random(), normalized=True)
)

incidents = gpd.GeoDataFrame(geometry=points, crs=edges.crs)
```

Since reported incidents affect not only the exact street where they occur but also the surrounding area, we design a penalty function based on distance. Edges within 100 meters of an incident receive a penalty that decreases linearly with distance. If an edge is influenced by multiple incidents, the penalties are cumulative.

```
python
edges['incident_penalty'] = 0.0

def incident_penalty(dist, limit=100):
    return max(0, 1 - dist/limit)

for pt in incidents.geometry:
    distances = edges.geometry.distance(pt)
    edges['incident_penalty'] += distances.apply(incident_penalty)
```

#### Safe spots

We include a positive input into to represent the presence of safe spots: local businesses, shops or institutions prepared to assist people experiencing harassment. These act as protective areas that reduce the perceived risk in their surroundings.

Safe spots work like the active incidents: we generate 15 randomly accross the map and will affect nearby edges using a linear decay function (the closer an edge is to a safe spot, the higher the bonus it receives).

Below you can see both Active incidents (in red) and Safe spots (in purple) represented in the same map:
 
<br /> <img width="2285" height="2520" alt="incidents_safespots (1)" src="https://github.com/user-attachments/assets/c29d4daf-d5eb-4f1c-885a-7ca5a2f3a05c" />

#### Historical risk 

Street Sexual Harassment often follows spatial patterns related to urban design, with certains areas being more prone to incidents. To represent this, we simulate a two-week period of incidents to capture recurring risks over both space and time.

We use a Poisson distribution to randomly generate incidents evey six hours across 14 days, with a higher rate of events during nighttime intervals.

```
python
# Define time bins: every 6 h over 14 days
start = datetime(2025, 6, 1)
timestamps = [start + timedelta(hours=6*i) for i in range(14*4)]

# Simulate “incidents” at random nodes per time bin
hist_points = []
hist_times  = []

for t in timestamps:
    # rate by hour
    lam = 22 if (t.hour >= 22 or t.hour < 6) else 15
    n = np.random.poisson(lam)
    if n == 0:
        continue

    # sample n segments (with replacement)
    segs = edges.sample(n=n,
                        replace=True,
                        weights=edges['length'])
    # pick a uniformly random point along each
    pts  = segs.geometry.apply(
        lambda line: line.interpolate(random.random(), normalized=True)
    )
    hist_points.extend(pts.tolist())
    hist_times .extend([t]*len(pts))

df_hist = gpd.GeoDataFrame(
    {'timestamp': hist_times},
    geometry=hist_points,
    crs=edges.crs
)
```

To reflect the fading influence of older incidents, we apply an exponential decay function based on recency. More recent events weigh more heavily, while older ones gradually lose importance.

```
python
# We assign each incident to its nearest node and compute the distance
df_hist = (
    gpd.sjoin_nearest(
        df_hist,
        nodes_gdf[['geometry']],
        how='left',
        distance_col='dist_to_node'
    )
    # after the join, the node ID is in osmid_right
    .rename(columns={'osmid': 'node'})
)


# Reference time: latest timestamp
ref = df_hist["timestamp"].max()

# Compute recency weights (exponential decay)
df_hist['weight'] = np.exp(
    -0.3 * ((ref - df_hist['timestamp']).dt.total_seconds() / 86400)
)

# Sum weights per node
node_risk = (df_hist
    .groupby('node')['weight']
    .sum()
    .reset_index(name='incident_weight')
)

# Normalize to [0,1]
node_risk["risk_score"] = MinMaxScaler()\
    .fit_transform(node_risk[["incident_weight"]])
```

This produces a historical intensity map, highlighting areas where incidents are more likely to recur:

<br /> <img width="2285" height="2540" alt="historical_intesnity (1)" src="https://github.com/user-attachments/assets/367cbc23-cb1a-4927-9304-883297f339bd" /> 

#### Combining inputs into the safety cost 

We weight the four factos according to their influence on perceived safety: 
* Length penalty: moderate importance.
* Active incident penalty: highest importance.
* Historical risk: significant importance.
* Safe spot bonus: bonus that reduces overall cost.

```
python
edges["combined_cost"] = (
    0.4 * edges["length_penalty"] +
    1.2 * edges["incident_penalty"] +
    0.8 * edges["historical_penalty"] -
    0.6 * edges["safe_bonus"]
).clip(lower=0.0)
```


This is the final map with the safety costs for each edge: 

<br /> <img width="2332" height="2520" alt="safety_cost" src="https://github.com/user-attachments/assets/6e6a703d-1816-4db7-a554-85362f1532dd" /> 

### Routing 
Now that we've computed the combined_costfor each segment, we can use the score to generate safe walking routes. 

We can identify routes that balance distance and perceived safety. This allows us to compare:

* The shortest route, which minimizes geographic distance and time, as in convencional apps.
* The safest route, which minimizes our custom safety cost.
* The balanced route, which is a middle-ground alternative that slightly increases travel distance but significantly proves safety perception.

#### Individual routing 

We will create a route example from Calle de San Bernardo, 105 to Calle de Ventura Rodríguez, 17.

First, we obtain the shortest and safest paths:

```
python
shortest = nx.shortest_path(G, orig_node, dest_node, weight='length')
safest   = nx.shortest_path(G, orig_node, dest_node, weight='combined_cost')
```

Users may be interested in a middle solution, a compromise between speed and safety rather than strictly minimizing one input (length or safety). To model this, we introduce a weighted combination of distance and safety cost, allowing flexible prioritization. In this example we use weights of 0.3 for distance and 0.7 for safety.

```
python
alpha = 0.3  
beta = 0.7 

edges['weight_cost'] = alpha*edges['len_s'] + beta*edges['cost_s']

nx.set_edge_attributes(
    G,
    values=edges['weight_cost'].to_dict(),
    name='weight_cost'
)

balanced_route = nx.shortest_path(G, source=orig_node, target=dest_node, weight='weight_cost')
```

The resulting map shows the three alternative routes:

<br /> <img width="2285" height="2490" alt="shortest_safest_balanced" src="https://github.com/user-attachments/assets/7d637844-3137-4831-b01e-cf7b9f7be2b9" /> 

We also test additional balanced routes with different safety weights: 10%, 40%, 60% and 90%, to observe how the routing behavior evolves as safety becomes a stronger priority.

<img width="2285" height="2505" alt="balanced_routes" src="https://github.com/user-attachments/assets/f1b91830-180e-4c67-8d60-da22d016bbc3" />

#### Large-Scale Testing (100 Origin-Destination Pairs)

To assess how the algorithm behaves across a variety of trips, we conduct a large-scale simulation with 100 randomly generated origin–destination (OD) pairs within the study area. This allows us to evaluate how safety-weighted routing performs in different spatial contexts, from short neighborhood walks to longer cross-district routes.

Each pair is generated by randomly selecting two points within accessible street segments. To ensure meaningful comparisons, we only keep routes with a minimum walking distance of 500 meters. For every OD pair, we calculate:

* Shortest route: baseline reference using geographic distance only.
* Safer alternatives: routes computed with different safety weights (40%, 60% and 100% safety emphasis).

This setup helps us understand how route characteristics evolve when users prioritize safety over efficiency.

```
python
pairs = []

while len(pairs) < 100:

    o_seg = candidates.sample(1).iloc[0].geometry
    d_seg = candidates.sample(1).iloc[0].geometry
  #Random point on segment
    o_pt  = o_seg.interpolate(random.random(), normalized=True)
    d_pt  = d_seg.interpolate(random.random(), normalized=True)
  #Snaps points to graph
    o_node = ox.nearest_nodes(G, X=o_pt.x, Y=o_pt.y)
    d_node = ox.nearest_nodes(G, X=d_pt.x, Y=d_pt.y)
    try:
      #checks a path exists ad calculates the shortest path length
        dist = nx.shortest_path_length(G, source=o_node, target=d_node, weight="length")
    except nx.NetworkXNoPath:
        continue

    # accept only if distance ≥ 500 m
    if dist >= 500:
        pairs.append({
            "orig_lat"    : o_pt.y,
            "orig_lon"    : o_pt.x,
            "dest_lat"    : d_pt.y,
            "dest_lon"    : d_pt.x,
            "route_length": dist
        })
```

To further analyze performance, we classify the 100 routes into three distance categories (short, medium and long), based on the 33rd and 66th percentiles of the shortest-route lengths. This categorization helps identify whether safety-based adjustments have a larger impact on shorter or longer trips.

```
python
def length_group(d):
    if d <= q33:   return "Short"
    elif d <= q66: return "Medium"
    else:          return "Long"


length_bins = pd.Series(short_dists.map(length_group),
                        name="length_group", index=pivot.index)

# Assign each pair the group it corresponds to
deltas_binned = deltas_df.merge(
    length_bins.reset_index(),
    on="pair_id",
    how="left"
)
```

By comparing average distance, estimated travel time, and total safety cost across all OD pairs and route types, we can quantify how much additional distance is typically required to achieve a safer path, providing insights into the trade-offs between efficiency and safety in pedestrian routing.

## Conclusions

This project demonstrates how routing algorithms can integrate safety as a core dimension of urban mobility. Although this work uses synthetic data, the methodology is designed to be scalable and adaptable to real-world data sources — such as crowdsourced incident reports, municipal safety records, or mobile app inputs.

The results show that routes optimized for safety are often only marginally longer than the shortest path but substantially reduce exposure to risky areas. This finding reinforces the importance of considering safety in urban routing, planning, and infrastructure design.

### Future improvements

* Incorporate real incident data and user reports.
* Extend the model to other neighborhoods and cities.
* Experiment with additional safety indicators, such as lighting or pedestrian density.

### Acknowledgments

This project was developed as part of the Master's Thesis in Computational Social Sciences program at Universidad Carlos III de Madrid, and builds upon the mission of B.MUUN, an application dedicated to improving women's safety in public spaces.
