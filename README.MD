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

[This section will outline the planned methodology for the project. It will describe the overall architecture of the system, including data collection, preprocessing, feature engineering, the selection and training of the machine learning model(s), model evaluation, and potentially the development of a web application. A flowchart illustrating the process from data acquisition to the web application interface will be included. The specific machine learning approach (e.g., regression techniques, neural networks) and the rationale behind its selection will be detailed here.]

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


## Data (Anticipated)

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