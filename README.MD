# ML ModClark IUH Parameter Estimation

## Problem Statement

Accurately estimating the parameters of the ModClark Instantaneous Unit Hydrograph (IUH) is crucial for effective hydrological modeling, impacting vital applications such as flood forecasting, water resource management, dam safety, and predicting streamflow responses to future land cover changes. Traditional methods for determining these parameters often rely heavily on historical observed rainfall-runoff data within a specific basin. This dependency poses significant challenges, particularly in ungauged basins where such data is scarce or entirely unavailable. Furthermore, even in gauged basins, the calibration processes can be time-consuming, and may not always yield optimal parameter sets that generalize well to different future conditions.

This project addresses these limitations by leveraging machine learning techniques to develop a model capable of estimating the ModClark IUH parameters (time of concentration and storage coefficient) based on readily available physiographic characteristics and storm events of a watershed. By establishing the relationship between these basin characteristics and the IUH parameters, the developed model aims to provide a more objective, efficient, and potentially more accurate approach to parameter estimation, especially in data-scarce regions. This work has the potential to significantly enhance the applicability and reliability of hydrological models across a wider range of scenarios and locations, particularly within the [**DEFINE STUDY REGION**], the intended study region for application and validation with hydrology practitioners.

## Instantaneous Unit Hydrograph (IUH) Concepts

The Instantaneous Unit Hydrograph (IUH) is a fundamental concept in hydrology that represents the theoretical direct runoff hydrograph resulting from a unit volume (e.g., 1 mm or 1 inch) of effective rainfall applied instantaneously and uniformly over the entire drainage basin. It essentially characterizes the basin's response to a unit impulse of rainfal, and conceptually, _it is unique for each watershed_.

Key assumptions underlying the IUH concept include:

* **Instantaneous Rainfall:** The effective rainfall occurs over an infinitesimally short period.
* **Uniform Rainfall:** The effective rainfall is distributed uniformly across the entire watershed.
* **Linearity:** The direct runoff response is directly proportional to the effective rainfall amount. Doubling the rainfall will double the runoff at every point in time.
* **Time-Invariance:** The basin's response characteristics remain constant over time for different rainfall events.

Therefore, the IUH serves as a building block for deriving the direct runoff hydrograph for any given effective rainfall hyetograph through convolution. **By knowing the IUH of a basin, we can predict its runoff response to more complex and realistic rainfall patterns.**


## The Modified Clark's IUH


The **Modified Clark (ModClark)** is a specific type of synthetic IUH. It is a modification of traditional Clark's IUH in which it explicitly accounts for variations in travel time to the watershed outlet using a gridded representation of the watershed to route excess precipitation to the subbasin outlet (as opposed to the traditional method that uses an equation to estimate the time-area histogram). In predicting runoff from complex storm events, it accounts for spatial variation of rainfall. Its key parameters are:

* **Travel of Concentration ($T_c$):** Represents the time (in hours) it takes for water to travel from the hydraulically most distant point in the watershed to the outlet. It reflects the basin's size and slope characteristics.

* **Storage Coefficient ($R$):** Represents the temporary storage (in hours) of water within the watershed as it travels towards the outlet. It reflects the influence of channel and floodplain storage. It assumes it is uniform throughout each grid cell.

* **Time-Area histogram:** Represents the watershed area that contributes to flow at the outlet as a function of time, due to translation effect. In hydrology, **translation** refers to the time delay associated with rainfall traveling from different parts of the watershed to the outlet. In ModClark, each grid cell has a different travel time to the outlet, which creates a **translation hydrograph** by shifting unit impulses over time.

Estimating these $T_{c}$ and $R$ parameters accurately is crucial for the ModClark model to effectively simulate the basin's runoff response. $T_c$ and $R$, are usually calibrated specifically for each watershed but, it heavily relies on observed data.**This project aims to develop a machine learning model to predict these parameters based on basin characteristics to be used for watershed where data is inexistent.**

More information about UH and Clark's IUH specifically may be found at the U.S. Army Corps of Engineers: [Unit Hydroggraph Concept](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/unit-hydrograph-basic-concepts), [Clark Unit Hydrograph Model](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/clark-unit-hydrograph-model),  [ModClark Model](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/modclark-model).

## Project Goals

This project aims to significantly enhance the efficiency and accessibility of hydrological modeling, particularly for engineering companies and government authorities involved in water resource management and flood prediction. The primary goals are:

* **Estimation of ModClark Parameters**: To develop a robust and accurate machine learning model capable of estimating the ModClark Instantaneous Unit Hydrograph (IUH) parameters directly from the geophysical characteristics of a watershed.

* **Development of a User-Friendly Web Application**: To create a simple-to-use web application that streamlines the process of hydrological analysis. This application will:
    * Enable users to delineate watershed boundaries by providing the outlet coordinates.

    * Automatically derive the necessary geophysical characteristics of the delineated watershed.

    * Employ the machine learning model to estimate the ModClark IUH parameters rapidly.

    * Provide functionality to simulate the watershed's flow response based on the estimated parameters and user-specified storm events.

## Methodology

This section outlines the methodology employed in this project to develop a machine learning model for estimating ModClark IUH parameters. The overall workflow, illustrated in a flowchart (link to be added later), encompasses data collection, preprocessing, feature engineering, machine learning model training, and evaluation.

### ModClark's IUH Computation

The Instantaneous Unit Hydrograph (IUH) describes how a watershed responds to a **single unit depth of rainfall** (e.g., 1 mm or 1 in) applied instantaneously and uniformly in space. The ModClark model uses a recursive routing approach to simulate how rainfall excess is translated through the travel-time distribution and then routed through a linear reservoir at the basin outlet. The IUH ordinates are obtained by the following equations:

$$IUH_{(i)} = c\,I_i + (1 - c)\,IUH_{(i-1)} \qquad (1)$$

$$c = \frac{\Delta t}{R + 0.5\,\Delta t} \qquad (2)$$

$$U_i = \frac{IUH_{(i-1)} + IUH_{(i)}}{2} \qquad (3)$$

where:

- $I_i$ [$T^{-1}$] is the inflow rate to the routing reservoir at time step $i$ (the ordinate of the translation hydrograph);
- $IUH_{(i)}$ [$T^{-1}$] is the instantaneous unit hydrograph ordinate at time step $i$;
- $U_i$ [$T^{-1}$] is the unit hydrograph ordinate at time step $i$;
- $c$ is the dimensionless routing coefficient;
- $\Delta t$ [$T$] is the routing time step;
- $R$ [$T$] is the storage coefficient of the linear reservoir.

Equation (1) is the recursive IUH routing equation, (2) defines the routing coefficient, and (3) computes the unit hydrograph ordinates by averaging consecutive IUH values.

#### Inflow Computation

The inflow rate $I_i$ for a unit pulse (1 mm or 1 in) is computed by:

$$I_i = \frac{A_i}{A\,\Delta t} \qquad (4)$$

where:

- $A_i$ [$L^2$] is the total watershed area contributing runoff in the interval $(i-1)\Delta t < t_k \le i\Delta t$;
- $A$ [$L^2$] is the total watershed area;
- $\Delta t$ [$T$] is the time step.

The contributing area $A_i$ is obtained by summing the areas of all grid cells $k$ whose travel time $t_k$ falls within the $i$th interval:

$$
A_i = \sum_{\substack{k\\(i-1)\Delta t < t_k \le i\Delta t}} A_k
\qquad (5)
$$

The travel time $t_k$ of each grid cell $k$ is derived from its travel distance $D_k$ by:

$$
t_k = T_c \,\frac{D_k}{D_{\max}} \qquad (6)
$$

where:

- $D_k$ [$L$] is the flow-path distance from cell $k$ to the watershed outlet;
- $D_{\max}$ [$L$] is the maximum travel distance within the watershed;
- $T_c$ [$T$] is the basin time of concentration.

#### Truncating the Infinite Tail

Because the recursive solution in (3) produces an infinitely long hydrograph tail, computation stops when the cumulative volume (area under the $U_i$ curve) exceeds **0.995** units of depth. The ordinates are then scaled so that the total volume equals exactly **1.0** unit of depth. The scaling factor is:

$$\text{Scaling Factor} = \frac{1.0}{\text{Cumulative Volume}}
\qquad (7)$$

The cumulative volume is approximated by trapezoidal integration:

$$\text{Cumulative Volume} = \sum_{i=1}^n \biggl(\frac{U_i + U_{i-1}}{2}\biggr)\,\Delta t
\qquad (8)$$

The adjusted $U_i$ is then obtained by:

 $$U_i^{\text{adj}} =  U_i \cdot \text {Scaling Factor} \qquad (9)$$

 where:
 - $U_i^{adj}$ is the adjusted unit hydrograph ordinate.

#### Real Storm Event

The convolution is performed using the scaled unit hydrograph ordinates $U_i$ from Equation (3) (after applying the scaling factor from Equations (7)–(8)):

$$
Q(t) = \sum_{\tau=0}^{t} P(\tau) \, U_{(t-\tau)}^{adj}\,\Delta t \qquad (10)
$$

where:
- $P(\tau)$ [$L/T$] is the rainfall excess intensity at time step $\tau$;
- $U_{(t-\tau)}^{adj}$ is the scaled unit hydrograph ordinate at lag $(t-\tau)$;
- $\Delta t$ is the hyetograph time interval;
- $Q(t)$ [$L/T$] is the resulting depth-rate hydrograph at time $t$.

To convert this depth-rate hydrograph to volumetric discharge $Q_{\mathrm{vol}}(t)$ [$L^3/T$], multiply by the watershed area $A$:

$$Q_{\mathrm{vol}}(t) = Q(t)\,\times\,A \qquad (11)$$

#### Step-by-Step Procedure Summary

The following step-by-step guide summarizes the procedures defined above, to go from data to ModClark's IUH to real storm events simulation:

1. Grid Preprocessing  
   - Input: Cell areas $A_k$ and travel distances $D_k$
   - Compute: Travel time for each cell using Equation (6): $t_k = T_c\,\frac{D_k}{D_{\max}}$

2. Compute Area Contribution per Time Step  
   - For each time interval $(i-1)\Delta t < t_k \leq i\Delta t$, sum the areas to compute $A_i$ using Equation (5).

3. Compute Inflow Rate  
   - Using Equation (4), compute inflow rate: $I_i = \frac{A_i}{A\,\Delta t}$

4. Compute Routing Coefficient
   - Compute routing coefficient $c$ using Equation (2): $c = \frac{\Delta t}{R + 0.5\,\Delta t}$

5. Compute IUH Recursively 
   - Use Equation (1) recursively to compute $IUH_{(i)}$: $IUH_{(i)} = c\,I_i + (1-c)\,IUH_{(i-1)}$

6. Compute Unit Hydrograph Ordinates 
   - Compute $U_i$ using Equation (3):  $U_i = \frac{IUH_{(i-1)} + IUH_{(i)}}{2}$

7. Truncate the Hydrograph and Normalize  
   - Compute cumulative volume using Equation (8): $
     \text{Cumulative Volume} = \sum_{i=1}^n \biggl(\frac{U_i + U_{i-1}}{2}\biggr)\,\Delta t$
   - Stop calculation when cumulative volume exceeds 99.5% of unit depth.
   - Apply scaling factor using Equation (7) to normalize hydrograph volume to 1.0 unit depth: $
     \text{Scaling Factor} = \frac{1.0}{\text{Cumulative Volume}}$

8. Simulate Streamflow for Real Storm Events 
   - Given precipitation excess $P(\tau)$ at each time step, compute discharge $Q(t)$ via convolution using Equation (10):  
     $Q(t) = \sum_{\tau=0}^{t} P(\tau)\,U_{(t-\tau)}^{adj}\,\Delta t$

9. Convert to Volumetric Discharge
   - Multiply depth-rate $Q(t)$ by watershed area $A$ to obtain volumetric discharge using Equation (11): $Q_{\mathrm{vol}}(t) = Q(t)\,\times\,A$

By following this procedure, you can compute the IUH and simulate streamflow hydrographs using the ModClark method.
                                                            
### Preprocessing

Common steps to preprocess the raw data are taken: time zone conversion to UTC, ~~fill in of gaps (based on the average)~~, coordinate reference system conversion and projection of geodata and filter data to the area of study and measurement unit conversions. After basic preprocessing, the following steps are taken to prepare the data:

#### Geomorphological Characteristics

Geomorphological characteristics quantify the physical properties of the drainage basin and its network, directly influencing how precipitation is transformed into runoff at the outlet, which is described by the IUH coefficients, Time of Concentration ($T_c$) and Storage Coefficient ($R$).

- The **Area** ($A$) of the watershed, defined as the total surface area draining to the outlet, is a fundamental control on runoff volume and timing. A larger basin area generally implies longer distances for water to travel from the most remote points to the outlet, directly increasing the Time of Concentration ($T_c$). Furthermore, larger basins typically possess a greater capacity for temporary water storage within their channels, floodplains, and the overall landscape, which tends to result in a larger Storage Coefficient ($R$).

- **Drainage Density** ($D_d$), calculated as the total length of all stream channels per unit area, is an indicator of how effectively the landscape is dissected by drainage pathways. A higher drainage density suggests a more developed network of channels that are closer together. This reduces the average distance water must travel as overland flow before entering the relatively faster channel network. Therefore, a higher $D_d$ typically results in a smaller $T_c$. The relationship with $R$ is more complex; while a dense network provides channel storage, the increased efficiency of drainage in basins with high $D_d$ often leads to a more rapid hydrological response, potentially indicating a smaller $R$ compared to basins with less developed networks where hillslope or non-channel storage might be more significant.

- The **Stream Length** ($L_{mc}$), specifically referring to the length of the main stream channel, represents a significant portion of the flow path through the basin's primary drainage route. A longer main channel increases the duration water spends within this concentrated flow path, directly contributing to a larger $T_c$. Similarly, the main channel provides a substantial volume for temporary storage during flood events. A longer $L_{mc}$ offers more linear space for channel storage, generally leading to a larger $R$.

- **Basin Length** ($L_b$), defined as the length of the longest flow path from the watershed outlet to the most hydrologically distant point on the drainage divide, is a critical measure of the basin's overall longitudinal extent. This parameter directly governs the maximum travel distance for water originating from the furthest reaches of the basin. As such, a larger $L_b$ is a primary control on the Time of Concentration ($T_c$), resulting in a larger value. While $L_b$ reflects the longest path where flow attenuation can occur, its direct influence on the Storage Coefficient ($R$) is generally less pronounced than its impact on $T_c$.

- The **Centroidal Flowpath** ($L_{ca}$) represents the length along the basin length ($L_b$) from the outlet to the point on $L_b$ nearest to the geographic centroid of the basin. This characteristic provides insight into the travel distance from a central, representative area of the watershed. It is often used in empirical formulations for $T_c$ and, similar to basin length, a larger $L_{ca}$ suggests longer travel times from the basin's central mass, contributing to a larger $T_c$.

- The **10-85 Flowpath** ($L_{10−85}$) is defined as the length along the basin length ($L_b$) spanning from a point 10% of the distance from the outlet to a point 85% of the distance from the outlet. This segment encompasses a significant and often representative portion of the basin's primary flow path. The length of this segment directly influences the travel time through a substantial middle part of the basin, meaning a larger $L_{10−85}$ will contribute to a larger $T_c$. The characteristics of the channel and valley along this segment can significantly influence temporary storage, impacting $R$.

- The **Main Channel Slope** ($S_mc$) quantifies the gradient of the primary drainage channel. A steeper main channel accelerates flow velocities within the most efficient part of the drainage network, leading to a substantial reduction in $T_c$. Furthermore, steeper channels typically offer less temporary storage volume per unit length compared to flatter channels. Consequently, a larger $S_{mc}$ tends to result in a smaller R.

- The **10-85 Slope** ($S_{10−85}$) is the slope calculated along the 10-85 flowpath segment. This parameter serves as a representative measure of the gradient over a critical portion of the basin's main flow path. A steeper slope along this segment implies faster flow conditions in a substantial part of the routing pathway, contributing to a smaller $T_c$. Similar to the main channel slope, steeper conditions along this representative segment limit the opportunity for temporary storage, leading to a smaller R.

 - The **Basin Slope** ($S_b$) is the average slope of the entire basin surface, derived from local elevation differences. This overall average steepness drives the velocity of overland flow before water enters channels. A larger $S_b$ leads to faster overland flow, which reduces the time it takes for water to reach the drainage network, thus decreasing $T_c$. Steeper overall slopes also minimize the potential for surface ponding and accelerate drainage towards channels, contributing to a smaller $R$.

- **Basin Relief** ($H$), representing the total elevation difference within the basin, is closely linked to overall steepness. Higher basin relief is generally associated with steeper slopes throughout the watershed, promoting faster flow velocities across the landscape. Therefore, a larger $H$ tends to decrease $T_c$. High relief terrain often has limited flat areas or extensive valleys suitable for significant temporary storage. As a result, a larger $H$ generally correlates with a smaller $R$.

- The **Compactness Coefficient** ($C_c$) is a shape index comparing the watershed's perimeter to that of a circle with the same area. Values closer to 1 indicate a more circular shape, while larger values suggest elongation or irregularity. More compact (circular) basins tend to have shorter average travel distances from the watershed boundary to the outlet compared to elongated basins of the same area. A smaller $C_c$ (indicating a more elongated shape) increases these travel distances, leading to a larger $T_c$. The shape also influences how flow converges at the outlet; more compact basins typically concentrate flow more rapidly, which might imply a smaller $R$ compared to elongated basins that spread the hydrograph over a longer period.

- The **Form Factor** ($R_f$) is another shape parameter, calculated as the ratio of basin area to the square of the basin length. Low values of $R_f$ characterize elongated basins, while higher values indicate more compact shapes. For a given area, a smaller $R_f$ means a larger basin length ($L_b$), which increases the travel time from the most distant point. Thus, a smaller $R_f$ tends to increase $T_c$. Like the compactness coefficient, the form factor relates to flow convergence patterns and can indirectly influence $R$.

- The **Elongation Ratio** ($R_e$) is a shape index given by the ratio of the diameter of a circle with the same area as the watershed to the basin length ($L_b$). Values closer to 1 represent more circular basins, and lower values indicate elongation. More elongated basins (smaller $R_e$) have a longer characteristic length relative to their width, increasing the time it takes for water to travel from the headwaters to the outlet. A smaller $R_e$ therefore tends to increase $T_c$. Elongation can also affect how flow is distributed and potentially stored within the basin, influencing $R$.

- The **Relief Ratio** ($R_h$), calculated as the ratio of total basin relief to basin length, is a direct measure of the overall steepness of the basin normalized by its length. A higher relief ratio signifies a steeper, more rugged basin. Steeper conditions accelerate flow velocities throughout the basin, resulting in a smaller $T_c$. Steeper basins generally have less capacity for storing water on the surface or in wide valleys, leading to a smaller $R$.

- The **Ruggedness Number** ($R_n$) is the product of basin relief and drainage density. This composite index reflects both the steepness and the density of the drainage network. A high ruggedness number indicates a basin with significant relief and a well-developed channel system, both of which promote rapid drainage. Consequently, a larger $R_n$ tends to decrease $T_c$. Highly rugged basins are typically efficient at routing water out quickly, with limited opportunities for prolonged storage in the steep, dissected terrain, leading to a smaller $R$.

- **Overland Flow Length** ($L_{of}$), defined as the average distance water travels over the land surface before entering a defined stream channel, is inversely related to drainage density. A larger average overland flow length means water spends more time moving slowly across the land surface before reaching the faster-flowing channel network. This increased time in the slower flow phase contributes to a larger $T_c$. Longer overland flow paths also offer more opportunity for infiltration and temporary surface storage, which could influence the basin's overall storage characteristics ($R$).

- **Channel Sinuosity** ($S_i$) is the ratio of the actual length of the main stream channel to the straight-line distance between its endpoints. It quantifies how winding or meandering the main channel is. A more sinuous channel (larger $S_i$) means water travels a longer distance within the channel network to reach the outlet compared to a straight path. This increased travel distance in the channel directly increases $T_c$. Meandering channels often develop in wider valleys with associated floodplains, providing increased capacity for temporary storage of water during higher flows. Therefore, a larger $S_i$ tends to increase $R$.

- Finally, **Land Cover** ($LC$) describes the type of vegetation and surface materials covering the watershed. Land cover profoundly influences hydrological processes such as infiltration, evapotranspiration, and surface roughness. Different land cover types (e.g., forest, urban, agriculture) have varying effects on the resistance to overland flow and the rate at which water infiltrates the soil, impacting the speed at which water reaches stream channels and moves through riparian areas, thus affecting $T_c$. Land cover also determines the potential for surface and subsurface storage; for instance, forests and grasslands promote infiltration and soil moisture storage, increasing the effective storage capacity ($R$), while urban areas with impervious surfaces significantly reduce infiltration and storage. Therefore, land cover is a crucial characteristic influencing both $T_c$ and $R$.

The table below summarizes the geomorphological characteristics above mentioned:

| Feature                     | Symbol      | Description                                                                                                                                                                   | Formula                                                                                                 |
| :-------------------------- | :---------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| Area                        | $A$         | Total area of the watershed.                                                                                                                                                  | Derived from DEM                                                                                        |
| Drainage Density            | $D_d$       | Total length of stream channels per unit area of the watershed.                                                                                                                | $D_d = \frac{\sum L_c}{A}$, where $\sum L_c$ is the total length of all stream channels in the basin    |
| Stream Length               | $L_{mc}$    | Length of the main stream channel.                                                                                                                                             | Derived from DEM                                                                                        |
| Basin Length                | $L_b$       | Length of the longest flow path within the basin.                                                                                                                             | Derived from DEM                                                                                        |
| Centroidal Flowpath         | $L_{ca}$    | Length along the basin length from the outlet to the point nearest to the basin centroid.                                                                                     | Derived from DEM                                                                                        |
| 10-85 Flowpath              | $L_{10-85}$ | Length along the basin length begining at 10% the basin length from the outlet to 85% the basin length from the outlet.                                                     | Derived from DEM                                                                           |
| Main Channel Slope          | $S_{mc}$    | Slope of the main stream channel.                                                                                                                                              | Derived from DEM                                                                                        |
| 10-85 Slope                 | $S_{10-85}$ | Slope of the 10-85 flowpath.                                                                                                                                              | Derived from DEM                                                                                        |
| Basin Slope                 | $S_b$       | Average slope of the entire basin. For each pixel it computes the maximum difference in elevation between its neighbours. The basin slope output is the average of all the computed slope values. | Derived from DEM                                                                                        |
| Basin Relief                | $H$         | Total difference in elevation within the basin.                                                                                                                                              | $H = \text{Max Elevation} - \text{Min Elevation}$, where Max Elevation is the highest elevation and Min Elevation is the minimum elevation in the basin |
| Compactness Coefficient     | $C_c$       | Ratio of the watershed perimeter to the circumference of a circle with the same area.                                                                                           | $C_c = \frac{P}{2\sqrt{\pi A}}$, where $P$ is the basin perimeter                                       |
| Form Factor                 | $R_f$       | Ratio of the watershed area to the square of the basin length.                                                                                                                  | $R_f = \frac{A}{L_b^2}$                                                                                 |
| Elongation Ratio            | $R_e$       | Ratio of the diameter of a circle with the same area as the watershed to the basin length.                                                                              | $R_e = \frac{2\sqrt{A/\pi}}{L_b}$                                                                       |
| Relief Ratio                | $R_h$       | Ratio of the total basin relief to the basin length.                                                                                                                   | $R_h = \frac{H}{L_b}$                                                                                   |
| Ruggedness Number           | $R_n$       | Product of the basin relief and the drainage density.                                                                                                                          | $R_n = H \times D_d$                                                                                    |
| Overland Flow Length        | $L_{of}$    | Average length of overland flow paths within the watershed.                                                                                                                    | $\approx \frac{1}{2D_d}$                                                                                |
| Channel Sinuosity           | $S_i$       | Ratio of the channel length to the straight-line distance between the channel's endpoints.                                                                                       | $S_i = \frac{L_{mc}}{L_{straight}}$, where $L_{straight}$ is the straight-line distance between the main channel's endpoints |
| Land Cover                  | $LC$        | Land cover type for the year of storm event identification, obtained from Multi-Resolution Land Characteristics (MRLC) data.                                                        | Derived from MRLC data                                                                                  |
---

#### Regulated Watersheds

Regulated watersheds (those containing dams, according to the National Inventory of Dams) will be excluded from the analysis, since it significantly alters $R$ and $Tc$.

#### Storm Event Selection

All storm events within a 10-year period of data will be considered for parameter optimization. A storm event is defined as a period of recorded rainfall depth at the gauging station that produces any amount of runoff. A storm event is considered to have ended if there are three consecutive time steps with zero flow recorded at the gauging station. To minimize the influence of snowmelt, events occurring in January and February will be filtered out. The following metadata and characteristics are recorded for each storm event (15-min resolution) and be used to assess $T_c$ and $R$ outliers after optmization: 

   * Total precipitation in the 14, 7, and 3 days prior to the storm start.
   * Total precipitation in the 24, 6, and 3 hours prior to the storm start.
   * Rainfall temporal variability.
   * Rainfall spatial variability.
   * Total precipitation depth.
   * Rainfall duration.
   * Month of occurrence.

#### Baseflow Separation

Baseflow separation is performed using the Recursive Digital Filter (RDF). Baseflow is the portion of streamflow that comes from subsurface sources, such as groundwater discharge. It is necessary to separate baseflow from the total streamflow to isolate the direct runoff component, which is the focus of IUH. The RDF method is chosen for its simplicity and effectiveness in separating baseflow from direct runoff. The RDF works by applying a filter to the streamflow data to partition it into baseflow and direct runoff components.

#### Voronoi Polygon

Voronoi polygons (also known as Thiessen polygons) are used to account for the spatial variation of rainfall. A Voronoi polygon is a region where any point inside the polygon is closer to a given precipitation station than to any other station. In this study, Voronoi polygons are used to assign a precipitation value to each grid cell (at 10m resolution) for the selected storm event. 

#### Precipitation Loss Method

Initial and constant loss method is used. Precipitation loss refers to the portion of rainfall that does not contribute to direct runoff due to processes such as infiltration, interception, and depression storage. This method is chosen for its simplicity and wide adoption in similar hydrological simulations. The initial loss represents the amount of precipitation that occurs before runoff begins, while the constant loss represents a continuous rate of loss throughout the storm event.

* **Parameter Optimization:** The ModClark parameters, $T_c$ and $R$, along with the initial and constant loss parameters, will be optimized concurrently. This is important because these parameters are often interdependent, and optimizing them together can lead to a more accurate representation of the rainfall-runoff process. The Differential Evolution (DE) algorithm will be used for this optimization. DE is a robust and efficient global optimization algorithm that is well-suited for this type of problem.

    Initial parameter values are set as follows: $T_c$ is estimated using the Kirpich equation (Patra, 2008), $R$ is initialized as (13/7) * $T_c$, initial loss is set to the total precipitation prior to the start of flow, and constant loss is initialized as the average hydraulic conductivity of the soil types in the watershed, weighted by their proportions.

    Reasonable bounds are applied: initial loss is constrained between 0 and the accumulated rainfall before the rising limb. Constant loss is bounded between 75% of the hydraulic conductivity of the least permeable soil and 125% of the hydraulic conductivity of the most permeable soil in the catchment. This approach is considered reasonable because it allows the parameters to vary within physically plausible ranges while still guiding the optimization process.

    The following table, reproduced from Rawls, Brakensiek, and Miller (1983), shows typical values of saturated hydraulic conductivity for different soil textures:

    | Soil Texture             | Saturated Hydraulic Conductivity (cm/hr) |
    | :----------------------- | :--------------------------------------- |
    | Sand                     | 29.9                                     |
    | Loamy Sand               | 14.7                                     |
    | Sandy Loam               | 11.2                                     |
    | Loam                     | 5.4                                      |
    | Silt Loam                | 6.8                                      |
    | Sandy Clay Loam          | 5.2                                      |
    | Clay Loam                | 2.6                                      |
    | Silty Clay Loam          | 2.0                                      |
    | Sandy Clay               | 1.5                                      |
    | Silty Clay               | 1.0                                      |
    | Clay                     | 0.6                                      |

    The Nash-Sutcliffe Efficiency (NSE) coefficient is used as the objective function for optimization. The NSE is defined as:

    $$
    NSE = 1 - \frac{\sum_{t=1}^{T} (Q_{obs}^t - Q_{sim}^t)^2}{\sum_{t=1}^{T} (Q_{obs}^t - \overline{Q}_{obs})^2}
    $$

    where:

    -   $Q_{obs}^t$ is the observed discharge at time step $t$,
    -   $Q_{sim}^t$ is the simulated discharge at time step $t$,
    -   $\overline{Q}_{obs}$ is the mean observed discharge over the entire period $T$.

    NSE measures the relative magnitude of the residual variance compared to the data variance. It ranges from -∞ to 1, with 1 indicating a perfect match between simulated and observed hydrographs. NSE is chosen because it is a widely used and robust metric for evaluating the overall fit of hydrological models.

    In addition to NSE, the following metrics are used to evaluate the optimized hydrographs:

    * Percentage difference in peak flow.
    * Difference in time to peak.
    * Difference in total volume

    The steps for parameter optimization are:
        1.  Define the objective function (NSE).
        2.  Set initial parameter values and bounds.
        3.  Run the Differential Evolution algorithm.
        4.  Evaluate the resulting hydrograph using NSE, percentage difference in peak flow, difference in time to peak and difference in total volume.
        5.  Repeat steps 3 and 4 until convergence criteria are met.

* **Filtering Outliers:** Theoretically, storm events within the same watershed should exhibit similar $T_c$ and $R$ values, as these parameters primarily depend on the watershed's physical characteristics, which are time-invariant. However, variations in estimated $T_c$ and $R$ for different storm events within the same watershed can arise due to errors in precipitation excess estimation (e.g., inaccuracies in estimating precipitation loss or neglecting spatial rainfall variability) or errors in baseflow separation. These errors can lead to inaccurate simulation of the direct runoff hydrograph, resulting in non-representative $T_c$ and $R$ values during the optimization process.

    Outliers in the optimized $T_c$ and $R$ values will be identified using the interquartile range (IQR) technique and excluded from further analysis. After filtering outliers, the $T_c$ and $R$ values for each watershed will be averaged to obtain representative values for that watershed. These averaged $T_c$ and $R$ values, along with the corresponding watershed characteristics, will then be used to train the machine learning models.

### Preparation for Machine Learning Algorithm

Watershed characteristics (geophysical and land cover) will be joined with the averaged $T_c$ and $R$ values for each watershed. Any additional data preparation steps required by the specific machine learning algorithms (e.g., scaling, normalization) will also be performed.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) will be conducted to understand the relationships between $T_c$, $R$, and the watershed characteristics. For multiple linear regression, EDA will help determine if any data transformations are needed to meet the model's assumptions. In general, EDA will be used to identify the most relevant watershed characteristics for predicting $T_c$ and $R$ across all the machine learning models.

### Machine Learning Model Training

The following machine learning algorithms will be tested:

* Multiple Linear Regression
* Random Forests
* Support Vector Machines (SVM)
* Light Gradient Boosting Machine (LightGBM)

Multiple linear regression will serve as a baseline model. Given the relatively small dataset, these models are expected to provide good performance. Random Forests and LightGBM are robust to outliers and can capture non-linear relationships. SVM can handle high-dimensional data effectively.

### Evaluation

The performance of the trained machine learning models will be evaluated using the following metrics:

* **Direct Parameter Evaluation:**
    * Root Mean Squared Error (RMSE) of $T_c$ and $R$.
    * Mean Absolute Percentage Error (MAPE) of $T_c$ and $R$.
    * Bias of $T_c$ and $R$.

* **Hydrograph Evaluation:** The models' ability to predict accurate hydrographs will be assessed by comparing simulated hydrographs against observed hydrographs from a set of storm events held out for evaluation. The following metrics will be used:
    * Nash-Sutcliffe Efficiency (NSE)
    * Percentage difference in peak flow.
    * Difference in time to peak.
    * Difference in total volume.

    These metrics provide a comprehensive evaluation of the models' performance, assessing both their accuracy in predicting $T_c$ and $R$ and their ability to reproduce observed streamflow hydrographs.


## Data (Anticipated)

All data used in this project are publicly available, sourced from various reputable sources:

| Data                                      | Source 

[This section will describe the types of data anticipated for use in this project. This may include:
* **Physiographic data:** Digital Elevation Models (DEMs) to derive basin morphometric characteristics (area, slope, drainage density, etc.).
* **Climatic data:** Rainfall data, temperature data, and potentially evapotranspiration data.
* **Hydrologic data (potentially for validation):** Streamflow data from gauging stations.
* **Soil data:** Information on soil types and properties within the basin.
* **Land cover data:** Information on vegetation and land use patterns.
The sources of this data (e.g., USGS, NASA, publicly available datasets) will also be mentioned.]

## Webapp Aplication

[This section will provide information on the web application that will be developed as part of this project. It will describe how users can interact with the application to input basin characteristics and obtain estimated ModClark IUH parameters. Details about the technology stack used for the web application and instructions for accessing it will be included here.]

## Usage

[This section will provide examples of how the developed machine learning model and the web application can be used. This might include steps for preparing input data, running the model or using the web interface, and interpreting the output (estimated $T_c$ and $R$ values).]

## License

[This section will include standard licensing information for the project, such as the MIT License or Apache License 2.0. This will specify the permissions and limitations for using, distributing, and modifying the code.]