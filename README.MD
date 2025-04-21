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

### ModClark's IUH Computation

The IUH describes how a watershed responds to a **single unit depth of rainfall (e.g., 1 mm or 1 in)** uniformly distributed in space and time. The ModClark model uses a recursive routing approach to simulate how rainfall excess is routed through each grid cell toward the watershed outlet. In this method, the IUH ordinates can be obtained by:


$$IUH_{(i)} = c I_i + (1 - c)IUH_{(i-1)} \qquad (1)$$
.
$$c = \frac{\Delta t}{R + 0.5\Delta t} \qquad (2)$$
.
$$U_i = \frac{IUH_{(i-1)} + IUH_{(i)}}{2} \qquad (3)$$  


Equations 1, 2 and 3 are known as the IUH Routing Equation, Routing Coefficient  and Unit Hydrograph Ordinates respectively, where $I_i \ [L\cdot{T^1}]$ is the ordinate of the translation hydrograph (which represents the inflow rate), $IUH_{(i)}$ is the instantaneous unit hydrograph.   $U_i\ [L\cdot{T^1}]$ is the unit hydrograph at time $i$. $c$ is the routing coefficient, $\Delta t$ is the time step used in routing, and $R$ is the storage coefficient.

$IUH_{(i)}$ is a recursive formula. The routing coefficient $c$ controls the degree of attenuation in the hydrograph and the unit hydrograph ordinates $U_i$ are computed by averaging consecutive IUH values.

The inflow rate at time $i$ can be obtained by:

$$I_i = \frac{A_i}{A \cdot \Delta t}\cdot P\qquad (4)$$

where $A_i$ is the watershed area contributing to inflow at time-step $i$, $A$ is the total watershed area and, $P$ is the single unit depth of rainfall excess (1 mm or 1 in). $A_i$ is obtained by summing the areas of all grid cells $k$ whose travel time $t_k$ satisfies $(i-1)\Delta t < t_k \leq i\Delta t$. Mathematically, this is expressed as:

$$A_i = \sum_{\substack{k \ (i-1)\Delta t < t_k \leq i\Delta t}} A_k \qquad (5)$$

where $A_k$ is the area of grid cell $k$, $t_k$ is the travel time of grid cell $k$. The travel time of each grid cell $k$ is obtained by:

$$T_{k} = T_{c}\frac{D_{k}}{D_{max}} \qquad (5)$$

where $D_{k}$ is the travel distance from grid cell $k$ to the watershed outlet and $D_{max}$ is the travel distance for the grid cell that is most distant from the watershed outlet. 

#### Truncating the infinitely Long Tail

As the solution in equation $(3)$ is recursive, outflow will theoretically continue for an infinite duration. Within this model, computation of the unit hydrograph ordinates continues until the volume of the outflow exceeds 0.995 inches or mm over watershed area. The hydrograph ordinates are then multiplied by a correction factor, so that the total volume becomes 1 unit depth. The scaling factor is obtained by:

$$\text {Scaling Factor} = \frac{1.0}{\text {Cumulative Volume}} \tag{6}$$ 

Since the hydrograph is discrete (values at intervals), the volume is approximated by summing the area of trapezoids under the curve:

$$\text {Cumulative Volume} = \sum \Big( \frac{U_i + U_{i-1}}{2} \Big) \cdot \Delta t \tag{7}$$


#### Unit Rainfall vs. Real Rainfall

- **Unit Rainfall Event**: One-time application of 1 mm of rain over all cells → Produces the **Unit Hydrograph**.
- **Actual Rainfall Event**: Varying depth and timing across cells → Requires **convolution** with the Unit Hydrograph to produce streamflow.

---

### 🌊 Convolution for Simulated Flow

Once the unit hydrograph \( U \) is constructed, we compute the simulated streamflow hydrograph by convolving it with the rainfall hyetograph \( P \):

$$
Q_i = \sum_{j=0}^{i} P_j \cdot U_{i-j} \qquad (4)
$$

Where:

- \( Q_i \): Simulated discharge at time \( i \) \((\text{m}^3/\text{s})\)
- \( P_j \): Rainfall excess at time \( j \) \((\text{mm} \cdot \text{km}^2)\)
- \( U_{i-j} \): Unit hydrograph ordinate \((\text{m}^3/\text{s})\)

---

### ✂️ Truncating the Infinitely Long Tail

The recursive nature of Equation (1) causes the hydrograph tail to **extend infinitely** due to the exponential decay. However, in practice, we truncate the IUH after its cumulative volume reaches **99.5% of the total volume**.

#### How to Truncate:

1. Compute cumulative sum of \( U_i \)
2. Find the time index where:
   $$
   \frac{\sum_{i=0}^{k} U_i}{\sum_{i=0}^{\infty} U_i} \geq 0.995
   $$
3. Truncate \( U \) beyond index \( k \)

4. Normalize \( U \) such that the total volume equals 1 unit of rainfall (1 mm over 1 km² = 1,000 m³):

   $$
   U_i^{\text{adjusted}} = \frac{U_i}{\sum_{j=0}^{k} U_j} \cdot 1{,}000 \qquad (\text{m}^3/\text{s})
   $$

---

### 🧭 Step-by-Step Procedure

1. **Grid Preprocessing**  
   - Input: Cell areas \( A_{\text{cell}} \, [\text{km}^2] \), travel times \( t_{\text{cell}} \, [\text{min}] \)

2. **Define Impulse Input (Unit Hydrograph Case)**  
   - Set: \( P = 1 \, \text{mm} \)  
   - For each cell, assign \( I_i = P \cdot A_{\text{cell}} \) at the corresponding travel time index

3. **Compute Routing Coefficient**  
   - Use Equation (2) with chosen \( R \, [\text{min}] \) and \( \Delta t \, [\text{min}] \)

4. **Route IUH**  
   - Use Equation (1) recursively for each time step

5. **Compute Unit Hydrograph Ordinates**  
   - Use Equation (3) to compute \( U_i \)

6. **Truncate the Hydrograph**  
   - Identify where cumulative volume ≥ 99.5%  
   - Apply normalization (see truncation section)

7. **Simulate Streamflow for Real Events**  
   - Given actual precipitation excess \( P_i \) per time step  
   - Use convolution via Equation (4) to compute \( Q_i \)

---

By following this procedure, you can compute the IUH and simulate streamflow hydrographs using the ModClark method, laying the foundation for robust runoff modeling and parameter optimization using machine learning techniques.


[This section will outline the planned methodology for the project. It will describe the overall architecture of the system, including data collection, preprocessing, feature engineering, the selection and training of the machine learning model(s), model evaluation, and potentially the development of a web application. A flowchart illustrating the process from data acquisition to the web application interface will be included. The specific machine learning approach (e.g., regression techniques, neural networks) and the rationale behind its selection will be detailed here.]

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