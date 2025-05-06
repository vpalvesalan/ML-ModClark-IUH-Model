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

Initial and $\text {Constant Loss}$ method is used. Precipitation loss refers to the portion of rainfall that does not contribute to direct runoff due to processes such as infiltration, interception, and depression storage. This method is chosen for its simplicity and wide adoption in similar hydrological simulations. The $\text {Initial Loss}$ represents the amount of precipitation that occurs before runoff begins, while the $\text {Constant Loss}$ represents a continuous rate of loss throughout the storm event.

#### Parameter Optimization

The parameters of the ModClark conceptual model, specifically the Time of Concentration ($T_c$), the Storage Coefficient ($R$), and the initial and $\text {Constant Loss}$ parameters, are optimized concurrently to calibrate the model against observed rainfall-runoff events. This concurrent optimization approach is critical due to the interdependence exhibited by these parameters during the calibration process. While the physical processes represented by these parameters – losses (such as infiltration and other abstractions) which determine the volume and timing of rainfall excess, and routing/storage ($T_c$ and $R$) which transform that excess rainfall into a simulated hydrograph at the outlet – are conceptually distinct and sequentially applied within the model structure, their influence on the final simulated hydrograph is cumulative. The observed hydrograph is the integrated result of both how much water becomes runoff and how it is subsequently transported and attenuated through the basin. Therefore, in the parameter space explored during calibration, adjustments to loss parameters to match observed runoff volumes or initiation times will impact the characteristics of the excess rainfall hydrograph that is routed. Similarly, modifications to routing parameters ($T_c$ and $R$) to improve the fit of the simulated hydrograph's timing and shape necessitate complementary adjustments in loss parameters to maintain or enhance the overall agreement with the observed flow. This complex interplay and mutual influence on the model's output hydrograph mean that the parameters are statistically and functionally interdependent during the optimization procedure, making their simultaneous calibration essential for obtaining a set of parameters that collectively provides the most accurate representation of the watershed's rainfall-runoff transformation.

*Algorithm*

The optimization process for calibrating the Time of Concentration ($T_c$), Storage Coefficient ($R$), Initial Loss, and $\text {Constant Loss}$ parameters will be performed using the **Differential Evolution (DE) algorithm**. DE is a powerful and widely-used global optimization algorithm that is particularly well-suited for complex, non-linear, and potentially multi-modal objective functions, characteristic of hydrological model calibration.

Unlike gradient-based optimization methods that rely on the derivative of the objective function and can easily get trapped in local minima, DE is a population-based stochastic search technique. It operates on a population of candidate solutions, iteratively improving them through a process inspired by natural evolution. Here's a breakdown of how it works:

- Initialization: The algorithm begins by generating an initial population of candidate solutions within the defined parameter bounds. Each solution is a vector containing values for the parameters being optimized ($T_c$, $R$, $\text {Initial Loss}$, and $\text  {Constant Loss}$). This initial population is typically generated randomly, ensuring a diverse exploration of the search space from the outset.

- Mutation: For each individual vector in the current population (target vector), DE creates a mutated vector. This is done by randomly selecting three distinct vectors from the population. The difference between two of these vectors is scaled by a mutation factor and added to the third vector. This "differential" strategy is a key feature of DE, allowing it to explore the search space effectively by leveraging the existing population's diversity.

 - Crossover: A trial vector is then generated by combining elements from the target vector and the mutated vector. A crossover probability determines, for each parameter, whether the value comes from the mutated vector or the original target vector. This step promotes the mixing of successful parameter combinations found in different individuals within the population.

 - Selection: The trial vector is then evaluated using the objective function - Nash-Sutcliffe efficiency ($NSE$), in this work. The trial vector is compared to the original target vector. If the trial vector yields a better objective function value (e.g., a lower error in hydrological simulation), it replaces the target vector in the next generation's population. Otherwise, the original target vector is retained.

This iterative process of mutation, crossover, and selection continues over many generations. The population of solutions evolves, with individuals progressively moving towards regions of the parameter space that yield better objective function values.

The calibration of hydrological model parameters presents several challenges that make DE a suitable choice:

- Non-linearity: The relationship between these parameters and the resulting hydrological response (e.g., streamflow hydrograph) is often highly non-linear. Small changes in one parameter can have complex and non-proportional effects on the output, making gradient-based methods less effective. DE's population-based approach and differential mutation are adept at navigating such non-linear landscapes.

- Parameter Interdependencies: The optimal value for one parameter can depend on the values of others. There are often complex interactions between $T_c$, $R$, and the loss parameters in determining the shape and timing of the simulated hydrograph. DE's ability to explore combinations of parameter values simultaneously across the population helps in identifying synergistic interactions that might be missed by methods optimizing parameters individually.

- Potential Multi-modality: The objective function for hydrological model calibration can have multiple local optima. This means there might be several different sets of parameter values that produce reasonably good fits to observed data, but only one set represents the globally optimal solution. DE's global search capabilities, driven by the diverse initial population and the mutation operation, reduce the risk of converging to a suboptimal local minimum.

- Robustness: DE is known for its robustness to the initial choice of parameters. Since it starts with a randomly generated population spanning the search space, it is less sensitive to the need for good initial guesses compared to local optimization methods.

- Simultaneous Optimization: DE naturally lends itself to the simultaneous optimization of multiple parameters, which is precisely what is required for calibrating $T_c$, $R$, $\text {Initial Loss}$, and $\text {Constant Loss}$ concurrently.

- Proven Application in Hydrology: Differential Evolution is known for its performance in hydrological models. Its effectiveness in handling the complexities inherent in hydrological optimization problems has been demonstrated in various research contexts. This track record provides confidence in its suitability for the current research.

_Initial Parameter Values_

Initial parameter values are set as follows: $T_c$ is estimated using the Kirpich equation (Patra, 2008): $T_c = 0.000323 \times L_b^{0.77} \times S_{10-85}^{-0.385}$. $R$ is initialized as $R = \frac {13}{7} \times T_c$. $\text {Initial Loss}$ is set to the total precipitation prior to the start of flow, and $\text {Constant Loss}$ is initialized as the average hydraulic conductivity of the soil types in the watershed, weighted by their proportions.

Reasonable bounds are applied: $\text {Initial Loss}$ is constrained between 0 and the accumulated rainfall before the start of runoff. $\text {Constant Loss}$ is bounded between 75% of the hydraulic conductivity of the least permeable soil and 125% of the hydraulic conductivity of the most permeable soil in the catchment. This approach is considered reasonable because it allows the parameters to vary within physically plausible ranges while still guiding the optimization process.

The following table, reproduced from Rawls, Brakensiek, and Miller (1983), shows typical values of saturated hydraulic conductivity for different soil textures:

| Soil Texture             | Saturated Hydraulic Conductivity (in/hr) |
| :----------------------- | :--------------------------------------- |
| Sand                     | 4.6|
| Loamy Sand               | 1.2|
| Sandy Loam               | 0.4|
| Loam                     | 0.1|
| Silt Loam                | 0.3|
| Sandy Clay Loam          | 0.06|
| Clay Loam                | 0.04|
| Silty Clay Loam          | 0.04|
| Sandy Clay               | 0.02|
| Silty Clay               | 0.02|
| Clay                     | 0.01|

_Objective Function_

The Nash-Sutcliffe Efficiency (NSE) coefficient is used as the objective function for optimization. The NSE is defined as:

$$NSE = 1 - \frac{\sum_{t=1}^{T} (Q_{obs}^t - Q_{sim}^t)^2}{\sum_{t=1}^{T} (Q_{obs}^t - \overline{Q}_{obs})^2}$$

where $Q_{obs}^t$ is the observed discharge at time step $t$, $Q_{sim}^t$ is the simulated discharge at time step $t$, $\overline{Q}_{obs}$ is the mean observed discharge over the entire period $T$.

NSE measures the relative magnitude of the residual variance compared to the data variance. It ranges from -∞ to 1, with 1 indicating a perfect match between simulated and observed hydrographs. NSE is chosen because it is a widely used and robust metric for evaluating the overall fit of hydrological models.

In addition to NSE, the following metrics are used to evaluate the optimized hydrographs:

   * Percentage difference in peak flow.
   * Difference in time to peak.
   * Difference in total volume

   The following steps summarizes the parameter optimization process:
   1.  Define the objective function (NSE).
   2.  Set initial parameter values and bounds.
   3.  Run the Differential Evolution algorithm.
   4.  Evaluate the resulting hydrograph using NSE
   5.  Repeat steps 3 and 4 until convergence criteria are met.

#### Outliers and data aggregation
Theoretically, storm events within the same watershed should exhibit similar $T_c$ and $R$ values, as these parameters primarily depend on the watershed's physical characteristics, which are time-invariant. However, variations in estimated $T_c$ and $R$ for different storm events within the same watershed can arise due to errors in precipitation excess estimation (e.g., inaccuracies in estimating precipitation loss or neglecting spatial rainfall variability) or errors in baseflow separation. These errors can lead to inaccurate simulation of the direct runoff hydrograph, resulting in non-representative $T_c$ and $R$ values during the optimization process. Therefore, errs in estimating any of the associated parameteres, is expected to result in outlier $T_c$ and $R$ parameteres.

Outliers will be identified using the interquartile range (IQR) technique and excluded from further analysis. After filtering outliers, the $T_c$ and $R$ values for each watershed will be averaged to obtain representative values for that watershed. These averaged $T_c$ and $R$ values, along with the corresponding watershed characteristics, will then be used to train the machine learning models.

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a critical initial phase of the modeling process, involving the systematic investigation of the dataset's main characteristics. In the context of this research, EDA will be conducted to gain a comprehensive understanding of the relationships between the geomorphological watershed characteristics and the target ModClark parameters, $T_c$ and $R$. This process involves summarizing the data's key properties through visual and statistical methods, identifying patterns, detecting potential anomalies or outliers, and informing subsequent steps in the modeling pipeline, including feature preprocessing and model selection.

A primary focus of EDA will be to examine the nature of the relationships between each independent variable (geomorphological feature) and the dependent variables ($T_c$ and $R$). This will involve generating scatter plots of each geomorphological feature against $T_c$ and against $R$. These visualizations are essential for visually assessing whether relationships appear linear, non-linear (e.g., curved, exponential, power-law), monotonic, or if there are indications of thresholds or sudden changes. For the linear models (OLS MLR and Elastic Net Regression), these scatter plots are particularly important for identifying features where transformations (such as logarithmic, power, or polynomial) may be necessary to approximate linearity, as discussed in the Feature Transformation and Scaling section. Additionally, examining residual plots after fitting preliminary linear models can help to diagnose patterns indicative of non-linearity or heteroscedasticity, further guiding transformation decisions.

Beyond assessing feature-target relationships, EDA will involve analyzing the characteristics of the geomorphological features themselves. Univariate analysis will include computing summary statistics (mean, median, standard deviation, quartiles, minimum, and maximum) and generating histograms or box plots for each feature. This provides insight into their central tendency, variability, range, and the shape of their distributions. Understanding feature distributions is valuable for identifying potential outliers and assessing the suitability of certain transformations (like Box-Cox, which aims to normalize distributions) or scaling methods.

Multivariate analysis will focus on understanding the relationships among the geomorphological features. A correlation matrix will be computed to quantify the pairwise linear association between features using Pearson correlation coefficients. If non-linear monotonic relationships are suspected, Spearman correlation coefficients will also be considered. Visualizations such as scatter plot matrices (for a subset of features) can provide a visual overview of these pairwise relationships. Identifying features that are highly correlated with each other (multicollinearity) is especially important, as it can impact the stability and interpretability of OLS MLR coefficient estimates. While Elastic Net is designed to handle multicollinearity through regularization, understanding these interdependencies among features is still valuable for interpreting the relative importance or roles of different geomorphological descriptors.

Furthermore, EDA will include examining the distributions and summary statistics of the target variables ($T_c$ and $R$) themselves. Understanding the range and variability of $T_c$ and $R$ within the dataset provides context for evaluating model performance. Data quality checks, such as identifying missing values, obvious errors, or physically impossible values (if not already addressed during initial data compilation), will also be conducted during the EDA phase to ensure the reliability of the dataset used for modeling.

In summary, EDA serves as a foundational step to thoroughly understand the dataset's structure, uncover relationships, identify data quality issues, and provide empirical grounding for subsequent decisions regarding feature preprocessing strategies and the selection, configuration, and interpretation of the chosen machine learning models.

### Machine Learning Model Training

To estimate the ModClark parameters, $T_c$, and $R$, from the delineated geomorphological characteristics, a suite of machine learning algorithms will be trained and evaluated. This selection aims to explore different modeling approaches, ranging from a fundamental linear model and its regularized extension to more complex non-linear ensemble methods, providing a comprehensive assessment of the relationship between watershed geomorphology and hydrological response parameters. The candidate models are **Multiple Linear Regression, Elastic Net Regression, Support Vector Machines (SVM)** and, **Random Forests**.

#### Multiple Linear Regression

Multiple Linear Regression (MLR) will be employed as a foundational model to serve as a baseline for comparison with more complex algorithms. MLR models the linear relationship between a dependent variable (in this case, either $T_c$ or $R$) and multiple independent variables (the geomorphological features). The general form of the model is expressed as:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

Here, $y$ represents the predicted parameter ($T_c$ or $R$), $x_1, x_2, \dots, x_p$ are the $p$ geomorphological features (e.g., Area, Basin Length, Drainage Density), $\beta_0$ is the intercept, $\beta_1, \beta_2, \dots, \beta_p$ are the coefficients representing the change in $y$ for a one-unit change in each respective feature $x_i$ (holding others constant), and $\epsilon$ is the error term.

Effective application of MLR relies on several key assumptions regarding the data and the error term:
* **Linearity:** There is a linear relationship between the independent variables ($x_i$) and the dependent variable ($y$).
* **Independence of Errors:** The errors ($\epsilon$) are independent of each other.
* **Homoscedasticity:** The variance of the errors is constant across all levels of the independent variables.
* **Normality of Errors:** The errors are normally distributed.
* **No Multicollinearity:** The independent variables are not highly correlated with each other.

Violations of these assumptions can affect the reliability and interpretability of the model's coefficients and statistical inferences.

Regarding data volume, MLR can provide reasonable performance and useful insights into linear relationships even with relatively small datasets, provided its assumptions are approximately met. However, its ability to capture complex, non-linear interactions between features is inherently limited.

MLR is not inherently sensitive to the scale of input features in terms of its predictive capability. The magnitude of the learned coefficients ($\beta_i$), however, is directly dependent on the scale of the corresponding feature ($x_i$). For example, if Area ($A$) is measured in km² versus m², the coefficient for Area will differ by a factor of $10^6$. MLR offers high interpretability, as the magnitude and sign of each coefficient $\beta_i$ directly indicate the estimated linear impact of the corresponding feature $x_i$ on the predicted output $y$, assuming the model assumptions are met.

#### Elastic Net Regression

Elastic Net Regression is a type of penalized linear regression that combines the penalties of both Lasso (L1 regularization) and Ridge (L2 regularization). This approach modifies the standard MLR objective function by adding a penalty term proportional to both the sum of the absolute values of the coefficients (L1) and the sum of the squared values of the coefficients (L2). The objective function minimized is:

$$\min_{\beta_0, \beta} \left\{ \frac{1}{N} \sum_{i=1}^N (y_i - (\beta_0 + x_i^T \beta))^2 + \lambda \left[ \alpha \|\beta\|_1 + (1-\alpha) \|\beta\|_2^2 \right] \right\}$$

Here, $N$ is the number of data points, $y_i$ is the observed value for the $i$-th data point, $x_i^T$ is the vector of features for the $i$-th data point, $\beta_0$ is the intercept, $\beta$ is the vector of coefficients, $\lambda$ is a non-negative regularization parameter controlling the overall strength of the penalty, and $\alpha$ is a parameter ($0 \le \alpha \le 1$) balancing the mix between the L1 penalty ($\|\beta\|_1$) and the L2 penalty ($\|\beta\|_2^2$). When $\alpha = 1$, Elastic Net becomes Lasso; when $\alpha = 0$, it becomes Ridge.

Elastic Net assumes a linear relationship between the features and the dependent variable, similar to OLS MLR. However, its estimation process does not strictly rely on the strong assumptions regarding the distribution, independence, and homoscedasticity of the error term in the same way OLS MLR does for statistical inference. A key advantage of Elastic Net over standard OLS MLR is its inherent capability to handle multicollinearity among features, which can be common among geomorphological parameters (e.g., Area and Basin Length, or different slope metrics). While OLS estimates become unstable and highly variable with correlated features, the L2 penalty component in Elastic Net effectively manages this issue by shrinking correlated coefficients. Furthermore, the combined L1 and L2 penalties make Elastic Net robust against overfitting, which is a significant concern when training models on datasets with a moderate number of data points relative to the number of features (like the 150-250 points and 18 features in this study). The L1 penalty also allows for feature selection, driving the coefficients of less relevant features exactly to zero, which can simplify the model and potentially highlight the most influential geomorphological predictors of $T_c$ and $R$.

Given the dataset size and the potential for correlated geomorphological features, Elastic Net is well-suited for this study. It provides a more robust and generalizable linear modeling approach compared to standard OLS MLR by mitigating the issues of multicollinearity and overfitting through regularization. It works effectively with moderate data volumes where more complex non-linear models might struggle to generalize without extensive tuning and validation.

Elastic Net, like other penalized linear models, is sensitive to the scale of input features. This is because the penalty terms are applied directly to the magnitude of the coefficient values. Features with larger numerical scales would see their corresponding coefficients penalized more heavily if the data is not scaled, potentially leading to suboptimal model fitting. Therefore, feature scaling (standardization or normalization) is essential before training an Elastic Net model to ensure that the regularization affects all coefficients proportionally based on their importance rather than their feature scale.

In terms of interpretability, Elastic Net offers a higher degree than more complex  models like Random Forests or non-linear SVMs. The learned coefficients can be inspected to understand the estimated (shrunk) linear impact of each feature. The feature selection property (when $\alpha > 0$) can further enhance interpretability by yielding sparser models that highlight a subset of features deemed most relevant by the model.

#### Support Vector Machines (SVM)

Support Vector Machines (SVM), specifically adapted for regression (Support Vector Regression - SVR), aim to find a function that deviates from the target by no more than a specified epsilon ($\epsilon$) for training data points, while being as flat as possible. SVR seeks to find a hyperplane in a high-dimensional space that best fits the data points. Using kernel functions (such as linear, polynomial, or Radial Basis Function - RBF), SVM can effectively model non-linear relationships by implicitly mapping the input features into a higher-dimensional space where a linear separation or fit is possible.

SVMs are particularly effective in high-dimensional spaces and can provide good generalization performance even on complex datasets. Their ability to handle non-linearities via kernels makes them suitable for potentially intricate relationships between geomorphological features and hydrological parameters.

SVR can perform well on moderate-sized datasets (like the 150-250 data points available), and its performance is often less affected by the number of features compared to some other algorithms when using appropriate kernels. However, selecting the optimal kernel and its parameters (hyperparameters) is crucial and can require significant tuning, which can be challenging with a limited number of data points.

Support Vector Machines, especially those employing distance-based kernels (like polynomial or RBF), are highly sensitive to the scale of input features. Features with larger numerical ranges can disproportionately influence the calculation of distances between data points, which is fundamental to defining the hyperplane and kernel computations. For instance, if Basin Area ($A$) values are orders of magnitude larger than Channel Sinuosity ($S_i$) values, Area could dominate the distance metric, reducing the effective contribution of Sinuosity to the model. Therefore, feature scaling (standardization or normalization) is essential when using SVMs to ensure that all features contribute appropriately to the model fitting process. Regarding interpretability, the degree varies with the kernel. An SVM with a linear kernel offers some interpretability, as the weights learned for each feature can be examined. However, when non-linear kernels (like RBF) are used, the model operates in a transformed, high-dimensional space, rendering it largely a "black box" where the relationship between input features and the output prediction is not directly human-interpretable.


#### Random Forests

Random Forests (RF) is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mean prediction of the individual trees (for regression). The algorithm introduces randomness through bootstrapping (sampling the training data with replacement for each tree) and random feature selection at each tree split, which helps to reduce variance and prevent overfitting.

RF models are powerful in their ability to implicitly capture complex, non-linear relationships and interactions between features without requiring explicit specification of these relationships. They are also known for their robustness to outliers in the input data and their capacity to handle different types of features.

Random Forests generally require more data than simple linear models to fully leverage their ability to learn complex patterns by building a diverse forest of trees. While they can perform reasonably well on moderate datasets (150-250 data points in this study's context), their performance typically improves with larger volumes of data. Compared to single decision trees, they are less prone to overfitting than single decision trees on small data due to the ensemble nature.

A significant advantage of tree-based models like Random Forests is that they are not sensitive to the scale of input features. The splitting decisions within each tree are based on comparisons of a feature's value to a threshold ($x_i \le threshold$). These comparisons are not affected by linear transformations (scaling) of the feature values. Therefore, feature scaling is not necessary for training Random Forest models. For example, whether Basin Length ($L_b$) is in meters or kilometers does not change the relative order of data points along that feature axis, which is what tree splits rely on. In terms of interpretability, Random Forests are often considered less interpretable than linear models. While it is possible to extract feature importance scores that rank features based on their overall contribution to the model's predictive power, understanding precisely how a specific combination of feature values for a given watershed leads to a particular prediction requires tracing paths through numerous individual trees, which is computationally intensive and difficult for human comprehension.

The selection of these diverse machine learning models allows for a robust investigation into the relationship between watershed geomorphological characteristics and ModClark parameters. By comparing the performance and insights gained from linear, regularized linear, kernel-based methods and, tree-based ensemble, this study aims to identify the most effective approach for estimating $T_c$ and $R$ from readily available watershed data.

### Feature Transformation and Scaling

The application of linear regression models, such as Ordinary Least Squares Multiple Linear Regression (OLS MLR) and Elastic Net Regression, relies on the assumption that the relationships between the independent variables (geomorphological features) and the dependent variables ($T_c$ and $R$) are linear. However, hydrological response parameters are often known to exhibit non-linear relationships with watershed characteristics, frequently described by power laws or other non-linear functions. Furthermore, algorithms like Elastic Net and SVM are sensitive to the scale and distribution of input features. To address these challenges and prepare the data for linear modeling, feature preprocessing steps involving transformation to promote linearity and scaling to standardize the feature ranges will be applied.

#### Feature Transformation for Linearity

To satisfy the linearity assumption of OLS MLR and Elastic Net, a systematic approach to feature transformation will be undertaken for each geomorphological characteristic. The objective is to transform the features such that their relationship with the target variables ($T_c$ and $R$) becomes approximately linear. The transformation process will be attempted in the following sequence for each feature:

1.  **Logarithmic Transformation:** This involves applying a logarithmic function (e.g., the natural logarithm) to the feature values ($x$). The transformation is $x' = \log(x)$. This method is particularly effective for features that have a skewed distribution (commonly right-skewed) or are believed to have a multiplicative or power-law relationship with the target variable, which is frequently the case for hydrological parameters related to basin size (Area, Lengths) and time. Log transformation linearizes relationships of the form $y = a \cdot x^b$ into $\log(y) = \log(a) + b \log(x)$, or $y = a \cdot b^x$ into $\log(y) = \log(a) + x \log(b)$. A prerequisite for the standard logarithmic transformation is that the feature values must be strictly positive.

2.  **Power Transformation (Box-Cox):** If a logarithmic transformation does not sufficiently linearize the relationship or normalize the feature's distribution, the Box-Cox transformation will be applied. The Box-Cox transformation is a family of power transformations parameterized by $\lambda$, designed to find the optimal power to transform a feature to be as close to a normal distribution as possible, which often aids in linearizing relationships in regression. The transformation is defined as:
    $$
    x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(x) & \text{if } \lambda = 0 \end{cases}
    $$
    The optimal value of $\lambda$ is estimated from the data. This transformation requires the input feature values to be strictly positive.

3.  **Polynomial Features:** If neither logarithmic nor Box-Cox transformations result in a sufficiently linear relationship, polynomial features will be considered as a final attempt at linearization for OLS MLR and Elastic Net. This involves creating new features by raising existing features to a power (e.g., $x^2, x^3$) or generating interaction terms between features (e.g., $x_1 x_2$). Including polynomial terms up to a certain degree allows the linear model to fit polynomial curves in the original feature space. However, given the dataset size (150-250 data points) and the initial number of features (18), the creation of polynomial features must be approached cautiously. The degree of polynomial expansion and the inclusion of interaction terms will be limited to avoid a drastic increase in the total number of features, which could jeopardize model efficiency and increase the risk of overfitting the limited training data, despite the regularization provided by Elastic Net.

Features for which none of these transformation methods successfully establish an approximately linear relationship with the target variables ($T_c$ and $R$) will be excluded from the input dataset used for training the OLS MLR and Elastic Net models.

#### Feature Scaling

Following feature transformation, the independent variables will be scaled to standardize their ranges ( in the case of of SVM model, it will be scaled directly. As SVM handles well complex relationship between dependent and independent variables, features will no be transformed prior to scaling when usded in SVM). This step is essential for algorithms like Elastic Net and Support Vector Machine, which are sensitive to the scale of input features due to the nature of their regularization penalties (L1 and L2 norms) or kernel calculations (distance metrics). Features with larger numerical values could otherwise disproportionately influence the model fitting process.

The chosen scaling method is **Z-score Standardization**. This method transforms each feature such that it has a mean of zero and a standard deviation of one. The formula for standardization is:

$$x' = \frac{x - \mu}{\sigma}$$

where $x$ is the original feature value, $\mu$ is the mean of the feature across the training dataset, and $\sigma$ is the standard deviation of the feature across the training dataset.

A point of consideration is that Z-score standardization can result in negative values for transformed features that originally represented physically non-negative quantities (e.g., Area, Length, Drainage Density). This occurs when a particular transformed feature value is less than the mean of that transformed feature across the dataset. It is important to clarify that these negative scaled values **do not imply that the original physical quantity is negative**. They are simply a mathematical consequence of centering the data around zero as part of the scaling process. These scaled values are numerical inputs used by the machine learning algorithms, which operate in a mathematical space where negative numbers are standard, and the sign indicates the value's position relative to the mean. This loss of direct physical interpretability in the scaled values is a necessary trade-off for preparing the data effectively for the chosen algorithms and is a standard practice in the field of machine learning. The mean and standard deviation used for scaling will be computed solely from the training data, and these same values will be used to transform the validation and test datasets to prevent data leakage.

### Evaluation

A rigorous evaluation framework is essential to assess the performance of the trained machine learning models in predicting the ModClark parameters ($T_c$ and $R$) and, ultimately, their utility in facilitating accurate hydrological simulation. The evaluation will be conducted on an independent dataset of watersheds and storm events that were not used during model training or validation. The evaluation process encompasses two main components: direct evaluation of the predicted parameters and assessment of the models' ability to reproduce observed hydrographs using the predicted parameters.

#### Direct Parameter Evaluation

This component focuses on quantifying the accuracy of the machine learning models in predicting the values of $T_c$ and $R$ directly based on the geomorphological features. For each predicted parameter ($T_c$ and $R$), the following metrics will be calculated on the independent test dataset:

* **Root Mean Squared Error (RMSE):** The RMSE measures the average magnitude of the errors between the predicted parameter values and the observed (or calibrated) parameter values. It is sensitive to large errors due to the squaring of differences. A lower RMSE indicates better predictive performance.
    $$
    \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}
    $$
    where $n$ is the number of observations in the test set (number of watersheds/events for which parameters are predicted), $y_i$ is the observed parameter value, and $\hat{y}_i$ is the predicted parameter value.

* **Mean Absolute Percentage Error (MAPE):** The MAPE measures the average of the absolute percentage errors. It provides a relative measure of error, making it useful for comparing performance across different parameters or datasets with varying scales.
    $$
    \text{MAPE} = \frac{1}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%
    $$
    A lower MAPE indicates better performance. This metric requires that the observed parameter values ($y_i$) are non-zero.

* **Bias:** The Bias measures the average difference between the predicted and observed parameter values. It indicates whether the model systematically over-predicts (positive bias) or under-predicts (negative bias) the parameter values across the test set.
    $$
    \text{Bias} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)
    $$
    A bias value close to zero indicates less systematic error in parameter prediction.

These metrics provide a quantitative assessment of the accuracy, magnitude of errors, and systematic deviations in the direct prediction of $T_c$ and $R$ values by each machine learning model on the independent test set.

#### Hydrograph Evaluation

Ultimately, the practical value of predicting $T_c$ and $R$ lies in their application within the Clark IUH model to simulate streamflow hydrographs. Therefore, a crucial part of the evaluation is to assess how well hydrographs simulated using the ML-predicted $T_c$ and $R$ values match observed hydrographs from independent storm events. For each watershed and associated storm event in the test dataset, the ModClark model will be run using the $T_c$ and $R$ values predicted by each trained ML model for that specific watershed. The resulting simulated hydrograph for each event will then be compared against the corresponding observed hydrograph using several widely accepted hydrological performance metrics, averaged across all test events ($N$):

* **Nash-Sutcliffe Efficiency (NSE):** The NSE is a widely used goodness-of-fit statistic that assesses the predictive power of hydrological models by comparing the model's performance to that of simply using the mean of the observed data. Calculated for each hydrograph comparison and then averaged across all test events. NSE ranges from $-\infty$ to 1, with NSE = 1 indicating a perfect match, NSE = 0 indicating the model's performance is no better than using the mean of the observed data, and NSE < 0 indicating the model performs worse than using the mean. The average NSE across events provides an overall measure of the models' hydrograph prediction skill.
    $$
    \text{Average NSE} = \frac{1}{N} \sum_{j=1}^N \left( 1 - \frac{\sum_{i=1}^{n_j} (Q_{obs,j,i} - Q_{sim,j,i})^2}{\sum_{i=1}^{n_j} (Q_{obs,j,i} - \bar{Q}_{obs,j})^2} \right)
    $$
    where $N$ is the number of storm events in the test set, $n_j$ is the number of time steps in the hydrograph for event $j$, $Q_{obs,j,i}$ and $Q_{sim,j,i}$ are the observed and simulated discharges at time step $i$ for event $j$, and $\bar{Q}_{obs,j}$ is the mean observed discharge for event $j$.

* **Mean Absolute Percentage Error (MAPE) of Peak Flow:** This metric quantifies the average relative error in the prediction of the peak discharge rate across all test storm events. It assesses the models' skill in predicting the magnitude of flood peaks.
    $$
    \text{MAPE}_{\text{Peak Flow}} = \frac{1}{N} \sum_{j=1}^N \left| \frac{Q_{obs,peak,j} - Q_{sim,peak,j}}{Q_{obs,peak,j}} \right| \times 100\%
    $$
    where $N$ is the number of test events, $Q_{obs,peak,j}$ is the observed peak discharge for event $j$, and $Q_{sim,peak,j}$ is the simulated peak discharge for event $j$.

* **Bias of Peak Flow:** This metric indicates whether the model systematically over-predicts (positive bias) or under-predicts (negative bias) the peak discharge across the test events.
    $$
    \text{Bias}_{\text{Peak Flow}} = \frac{1}{N} \sum_{j=1}^N (Q_{sim,peak,j} - Q_{obs,peak,j})
    $$

* **Mean Absolute Error (MAE) of Time to Peak:** This metric quantifies the average magnitude of the error in predicting the timing of the maximum discharge across all test storm events. Accurate timing is crucial for flood forecasting and warning systems.
    $$
    \text{MAE}_{\text{Time to Peak}} = \frac{1}{N} \sum_{j=1}^N |T_{obs,peak,j} - T_{sim,peak,j}|
    $$
    where $N$ is the number of test events, $T_{obs,peak,j}$ is the observed time of peak for event $j$, and $T_{sim,peak,j}$ is the simulated time of peak for event $j$.

* **Bias of Time to Peak:** This metric indicates whether the model systematically predicts the time to peak earlier (negative bias) or later (positive bias) than observed, on average, across the test events.
    $$
    \text{Bias}_{\text{Time to Peak}} = \frac{1}{N} \sum_{j=1}^N (T_{sim,peak,j} - T_{obs,peak,j})
    $$

* **Mean Absolute Percentage Error (MAPE) of Total Volume:** This metric quantifies the average relative error in the total runoff volume simulated for each storm event across the test set. It assesses the models' skill in maintaining water balance.
    $$
    \text{MAPE}_{\text{Volume}} = \frac{1}{N} \sum_{j=1}^N \left| \frac{V_{obs,j} - V_{sim,j}}{V_{obs,j}} \right| \times 100\%
    $$
    where $N$ is the number of test events, $V_{obs,j}$ is the observed total runoff volume for event $j$, and $V_{sim,j}$ is the simulated total runoff volume for event $j$.

* **Bias of Total Volume:** This metric indicates whether the model systematically over-predicts (positive bias) or under-predicts (negative bias) the total runoff volume across the test events.
    $$
    \text{Bias}_{\text{Volume}} = \frac{1}{N} \sum_{j=1}^N (V_{sim,j} - V_{obs,j})
    $$

By combining the direct evaluation of predicted $T_c$ and $R$ values with the assessment of the resulting simulated hydrographs against observed data, this evaluation framework provides a comprehensive understanding of each machine learning model's performance. It evaluates both the accuracy of the parameter predictions and their ultimate utility in reproducing key features of the complex rainfall-runoff process (overall fit, peak flow magnitude, timing, and volume) within the ModClark conceptual framework. This dual approach ensures that the selected model is not only capable of estimating parameters but also enables hydrologically plausible and accurate simulations of streamflow.

## Data

All data utilized in this project are publicly available, sourced from various reputable institutions and repositories. The primary datasets acquired and their sources, formats, and intended uses are summarized in table below.

| Data                              | Source                                                                                                                                                      | Format     | Description                                                                                                     | Documentation Link                                                                                                                                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| US Boundary - States              | [US Census Bureau TIGER/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=States+%28and+equivalent%29)                    | `.shp`     | State boundaries visualization                                                                                 | [TechDoc](https://www2.census.gov/geo/pdfs/maps-data/data/tiger/tgrshp2024/TGRSHP2024_TechDoc.pdf)                                                          |
| US Boundary - Counties            | [US Census Bureau TIGER/Line](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=Counties+%28and+equivalent%29)                  | `.shp`     | County boundaries visualization                                                                                | –                                                                                                                                                            |
| Watershed Boundary      | [USGS TNM API](https://tnmaccess.nationalmap.gov/api/v1/docs)                                                                                              | `.shp`     | Delimit study region                                                                                           | [ScienceBase](https://www.sciencebase.gov/catalog/item/5a1632b3e4b09fc93dd171de)                                                                             |
| Stream Gauge Stations            | [USGS](https://www.sciencebase.gov/catalog/item/63140610d34e36012efa3836)                                                                                   | `.shp`     | Stream gauge locations                                                                                         | –                                                                                                                                                            |
| Precipitation Gauge Stations     | [NOAA - NCEI](https://gis.ncdc.noaa.gov/kml/precip_15.kmz)                                                                                                  | `.kmz`     | Precipitation station locations                                                                                | –                                                                                                                                                            |
| Dam Location                     | [USACE NID](https://nid.sec.usace.army.mil/#/downloads)                                                                                                     | `.gpkg`    | Identify regulated watershed flow                                                                             | [Resilience Dataset](https://resilience.climate.gov/datasets/fedmaps::national-inventory-of-dams-1/about)                                                   |
| World UTM Grid                   | [Esri](https://hub.arcgis.com/datasets/esri::world-utm-grid/explore)                                                                                       | `.GeoJSON` | Identify UTM zone                                                                                              | –                                                                                                                                                            |
| Digital Elevation Model (DEM)    | [USGS Earth Explorer](https://earthexplorer.usgs.gov/)                                                                                                      | `.tiff`    | Watershed delineation & geomorphology                                                                         | [SRTM Doc](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-void)                        |
| Water Body Raster                | [USGS](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-water)                              | `.tiff`    | Calculate water body percentages                                                                              | –                                                                                                                                                            |
| Historical Land Cover    | [MRLC Viewer](https://www.mrlc.gov/viewer/)                                                                                                                 | `.GeoTIFF` | Extract land use features                                                                                      | [Doc](https://www.mrlc.gov/documentation)                                                                                                                    |
| Streamflow Data                  | [USGS dataRetrieval](https://doi-usgs.github.io/dataRetrieval/)                                                                                             | `.json`    | Streamflow data for modeling                                                                                   | [Doc](https://doi-usgs.github.io/dataRetrieval/)                                                                                                                                                            |
| Precipitation Data          | [NOAA FTP](https://www1.ncdc.noaa.gov/pub/data/hpd/auto/v2/beta/15min/)                                                                                     | `.csv`     | Storm event selection                                                                                          | [Readme](https://www1.ncdc.noaa.gov/pub/data/hpd/auto/v2/beta/15min/readme.15min.txt)                                                                       |
| Soil Texture          | [ISRIC World Soil Texture Classifications](https://www.soilgrids.org/)                                                                                     | `.tif`     | Initiate parameters optmization                                                                                         | [Doc](https://www.earthdata.nasa.gov/dashboard/data-catalog/soil-texture)                                                                       |

These datasets were acquired and processed to provide the necessary inputs for deriving watershed geomorphological characteristics and for hydrological model calibration and evaluation. They can be broadly categorized as follows:

**Physiographic Data:** Digital Elevation Models (DEMs) form the core of the physiographic data, obtained from the U.S. Geological Survey (USGS) Earth Explorer. These raster datasets, typically in `.tiff` format, are fundamental for accurately delineating watershed boundaries and calculating a comprehensive suite of terrain-based morphometric parameters essential for characterizing basin form and drainage patterns. This includes properties like basin area, various slope indices (e.g., average watershed slope, channel slopes), basin relief, basin length (longest flow path), stream network metrics (e.g., stream length, drainage density), and shape factors (e.g., compactness coefficient, form factor, elongation ratio, ruggedness number). 

**Boundary and Point Data**: Boundary files for states and counties from the US Census Bureau TIGER/Line® and watershed boundaries from the USGS National Map API, provided in `.shp` format, serve as crucial references for study region definition, and visualization. A World UTM Grid from [Esri](https://hub.arcgis.com/datasets/esri::world-utm-grid/explore) in `.GeoJSON` format was used for identifying appropriate spatial reference systems to ensure geometric accuracy. Dam location data from the USACE National Inventory of Dams (NID), provided as `.gpkg` files, were consulted to identify and potentially exclude watersheds with significant upstream flow regulation, ensuring the applicability of the Clark IUH conceptualization which is typically applied to unregulated basins. Stream gauge station locations from [USGS](https://www.sciencebase.gov/catalog/item/63140610d34e36012efa3836) provide the points at which streamflow is monitored and the outflow point at which the watershed is delineated. Precipitation gauge station locations, obtained from NOAA - NCEI ([https://gis.ncdc.noaa.gov/kml/precip_15.kmz](https://gis.ncdc.noaa.gov/kml/precip_15.kmz)), identify meteorological monitoring sites.

**Hydrologic Data:** Hydro-meteorological observations are indispensable for calibrating and evaluating hydrological models. Historical streamflow time series data for the chosen stations were retrieved using the USGS dataRetrieval package, providing the observed hydrographs in `.json` format that serve as the target for model calibration and the benchmark for hydrograph evaluation.  Corresponding historical precipitation intensity data, such as 15-minute records accessed from NOAA FTP servers ([https://www1.ncdc.noaa.gov/pub/data/hpd/auto/v2/beta/15min/](https://www1.ncdc.noaa.gov/pub/data/hpd/auto/v2/beta/15min/)) in `.csv` format, are used for selecting significant storm events and providing the precipitation input to the ModClark hydrological model simulations.

**Soil Data:** Information on soil properties is relevant for characterizing the infiltration and storage capacity of the watershed surface and subsurface. Soil texture data, was sourced from the ISRIC World Soil Texture Classifications ([https://www.soilgrids.org/](https://www.soilgrids.org/)) provided in `.tif` format, offering spatial information on soil composition. This data is utilized to inform initial estimates or refine the range of values for soil-related parameters, such as the constant precipitation loss rate, during the ModClark model optimization process.

**Land Cover Data:** Land cover and land use patterns within a watershed significantly influence hydrological processes by affecting surface roughness, infiltration rates, evapotranspiration, and surface storage. Historical Land Cover data, such as the Multi-Resolution Land Characteristics (MRLC) data ([https://www.mrlc.gov/viewer/](https://www.mrlc.gov/viewer/)), typically available in `.GeoTIFF` format, is employed to extract land use features. This information is crucial for understanding how human activities and natural vegetation patterns impact runoff generation and is incorporated directly as input features for the machine learning models.

The use of publicly available datasets from authoritative sources enhances the reproducibility of this research. The integration of these diverse data types provides the foundation for characterizing the study watersheds and supports the development and evaluation of machine learning models for hydrological parameter estimation.


## License

[This section will include standard licensing information for the project, such as the MIT License or Apache License 2.0. This will specify the permissions and limitations for using, distributing, and modifying the code.]