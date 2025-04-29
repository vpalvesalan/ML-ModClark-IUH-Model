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

#### Step-by-Step Procedure

The following step-by-step guide summarizes the procedures defined above:

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

Common steps to preprocess the raw data are taken: time zone conversion to UTC, fill in of gaps (based on the average), filter geodata to the area of study, delineate watershed boundary and derive characteristics. After basic preprocessing, the following steps are taken to prepare the data:

* **Voronoi Polygon:** Voronoi polygons are used to account for the spatial variation of rainfall. A Voronoi polygon is a region where any point inside the polygon is closer to a given precipitation station than to any other station. In this study, Voronoi polygons are used to assign a precipitation value to each grid cell (at 10m resolution) for the selected storm event. This method is also used to calculate the average precipitation over the watershed for storm event selection.

* **Watershed Delineation:** Watershed boundaries and their characteristics (flow direction, geophysical characteristics, and cell distance to the outlet) are derived from Digital Elevation Models (DEMs). The table below lists the geophysical features derived:

    | Feature                     | Description                                                                                                                                                                   |
    | :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | Area                        | Total area of the watershed.                                                                                                                                                  |
    | Slope                       | Average slope of the watershed.                                                                                                                                                 |
    | Drainage Density            | Total length of stream channels per unit area of the watershed.                                                                                                                |
    | Stream Length               | Length of the main stream channel.                                                                                                                                             |
    | Basin Length                | Maximum distance from the watershed outlet to the watershed boundary.                                                                                                         |
    | Main Channel Slope          | Slope of the main stream channel.                                                                                                                                              |
    | Hypsometric Integral        | A measure of the distribution of land elevation within a watershed.                                                                                                            |
    | Compactness Coefficient     | Ratio of the watershed perimeter to the circumference of a circle with the same area.                                                                                           |
    | Form Factor                 | Ratio of the watershed area to the square of the basin length.                                                                                                                  |
    | Elongation Ratio            | Ratio of the diameter of a circle with the same area as the watershed to the maximum basin length.                                                                              |
    | Relief Ratio                  | Ratio of the total basin relief to the maximum basin length.                                                                                                                   |
    | Ruggedness Number             | Product of the basin relief and the drainage density.                                                                                                                          |
    | Overland Flow Length        | Average length of overland flow paths within the watershed.                                                                                                                    |
    | Channel Sinuosity           | Ratio of the channel length to the straight-line distance between the channel's endpoints.                                                                                       |
    | Land Cover                  | Land cover type for the year of storm event identification, obtained from Multi-Resolution Land Characteristics (MRLC) data.                                                        |

    Regulated watersheds (those containing dams, according to the National Inventory of Dams) will be excluded from the analysis.

* **Storm Event Selection:** All storm events within a 10-year period of data will be considered for parameter optimization. A storm event is defined as a period of recorded rainfall depth at the gauging station that produces any amount of runoff. A storm event is considered to have ended if there are three consecutive time steps with zero flow recorded at the gauging station. To minimize the influence of snowmelt, events occurring in January and February will be filtered out. The following metadata and characteristics are recorded for each storm event (15-min resolution):

    * Total precipitation in the 14, 7, and 3 days prior to the storm start.
    * Total precipitation in the 24, 6, and 3 hours prior to the storm start.
    * Rainfall temporal variability.
    * Rainfall spatial variability.
    * Total precipitation depth.
    * Rainfall duration.
    * Month of occurrence.

* **Baseflow Separation:** Baseflow separation is performed using the Recursive Digital Filter (RDF). Baseflow is the portion of streamflow that comes from subsurface sources, such as groundwater discharge. It is necessary to separate baseflow from the total streamflow to isolate the direct runoff component, which is the focus of the ModClark model. The RDF method is chosen for its simplicity and effectiveness in separating baseflow from direct runoff. The RDF works by applying a filter to the streamflow data to partition it into baseflow and direct runoff components.

* **Precipitation Loss Method:** Initial and constant loss method is used. Precipitation loss refers to the portion of rainfall that does not contribute to direct runoff due to processes such as infiltration, interception, and depression storage. This method is chosen for its simplicity and wide adoption in similar hydrological simulations. The initial loss represents the amount of precipitation that occurs before runoff begins, while the constant loss represents a continuous rate of loss throughout the storm event.

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