# ML ModClark IUH Parameter Estimation

## Problem Statement

Accurately estimating the parameters of the ModClark Instantaneous Unit Hydrograph (IUH) is crucial for effective hydrological modeling, impacting vital applications such as flood forecasting, water resource management, dam safety, and predicting streamflow responses to future land cover changes. Traditional methods for determining these parameters often rely heavily on historical observed rainfall-runoff data within a specific basin. This dependency poses significant challenges, particularly in ungauged basins where such data is scarce or entirely unavailable. Furthermore, even in gauged basins, the calibration processes can be time-consuming, and may not always yield optimal parameter sets that generalize well to different to future conditions.

This project addresses these limitations by leveraging machine learning techniques to develop a model capable of estimating the ModClark IUH parameters (time of concentration and storage coefficient) based on readily available physiographic characteristics precipitation events of a watershed. By establishing the relationship between these basin characteristics and the IUH parameters, the developed model aims to provide a more objective, efficient, and potentially more accurate approach to parameter estimation, especially in data-scarce regions. This work has the potential to significantly enhance the applicability and reliability of hydrological models across a wider range of scenarios and locations, particularly within the Hudson River Basin, the intended study region for application and validation with hydrology practitioners.

## Instantaneous Unit Hydrograph (IUH) Concepts

The Instantaneous Unit Hydrograph (IUH) is a fundamental concept in hydrology that represents the theoretical direct runoff hydrograph resulting from a unit volume (e.g., 1 mm or 1 inch) of effective rainfall applied instantaneously and uniformly over the entire drainage basin. It essentially characterizes the basin's response to a unit impulse of rainfall.

Key assumptions underlying the IUH concept include:

* **Instantaneous Rainfall:** The effective rainfall occurs over an infinitesimally short period.
* **Uniform Rainfall:** The effective rainfall is distributed uniformly across the entire watershed.
* **Linearity:** The direct runoff response is directly proportional to the effective rainfall amount. Doubling the rainfall will double the runoff at every point in time.
* **Time-Invariance:** The basin's response characteristics remain constant over time for different rainfall events.

The IUH serves as a building block for deriving the direct runoff hydrograph for any given effective rainfall hyetograph through convolution. **By knowing the IUH of a basin, we can predict its runoff response to more complex and realistic rainfall patterns.**

The **ModClark IUH** is a specific type of synthetic IUH that utilizes a combination of a geomorphological (Clark) model and a time-delay function to represent the translation and attenuation of runoff within a watershed. Its key parameters are:

* **Time of Concentration ($T_c$):** Represents the time it takes for water to travel from the hydraulically most distant point in the watershed to the outlet. It reflects the basin's size and slope characteristics.
* **Storage Coefficient ($R$):** Represents the temporary storage of water within the watershed as it travels towards the outlet. It reflects the influence of channel and floodplain storage.

Estimating these $T_c$ and $R$ parameters accurately is crucial for the ModClark model to effectively simulate the basin's runoff response. **This project aims to develop a machine learning model to predict these parameters based on basin characteristics.**

More information may be found at the U.S. Army Corps of Engineers: [Unit Hydroggraph Concept](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/unit-hydrograph-basic-concepts), [Clark Unit Hydrograph Model](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/clark-unit-hydrograph-model),  [ModClark Model](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/transform/modclark-model).

## Project Goals

[This section will detail the specific goals of the project, including the primary objective of developing a machine learning model for ModClark IUH parameter estimation. It will also specify the study region (Hudson River Basin) and the intended outcomes, such as demonstrating the model's performance and potential benefits for hydrological applications relevant to engineering companies and government authorities.]

## Methodology (Planned)

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