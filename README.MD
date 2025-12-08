# ML ModClark IUH Parameter Estimation

[![Project Status](https://img.shields.io/badge/Status-Completed-success)](README.md)

## 💡 Project Summary

This project investigates the feasibility of using **Machine Learning (ML)** models to predict the two critical parameters of the **Modified Clark Instantaneous Unit Hydrograph (ModClark IUH)** model—the **Time of Concentration ($T_c$)** and the **Storage Coefficient ($R$)**—for ungauged watersheds in the **Ohio River Basin Region**.

The core challenge addressed is the regionalization of event-based hydrological parameters using only static geomorphological and land use characteristics.

## 🎯 Key Findings & Conclusions

The results of the comparative ML analysis (Elastic Net, Random Forest, SVM) revealed a stark difference in parameter predictability:

* **Time of Concentration ($T_c$):** Demonstrated a modest but statistically significant predictive relationship with static features. The best model (Elastic Net) achieved an **$R^2$ of $\approx 0.35$**.
* **Storage Coefficient ($R$):** **Cannot be reliably predicted** from static geomorphological features alone. All models failed to find a meaningful relationship, yielding an $R^2$ near zero ($\approx 0.06$).
* **Hydrologic Validation:** Simulations using ML-predicted parameters resulted in **catastrophic model failure** (astronomically negative average NSE), definitively isolating the **inability to predict $R$** as the primary source of error.
* **Conclusion:** The storage parameter ($R$) is dominated by **dynamic, event-specific variables** (e.g., antecedent moisture, storm intensity), highlighting the limitations of static regionalization approaches for event-based storage components.

## 🛠️ Methodology & Workflow

The project followed a rigorous, reproducible data science workflow:

1.  **Data Curation:** Automated screening of stream gauge stations and acquisition of high-resolution DEMs, precipitation, and streamflow data.
2.  **Geospatial Preprocessing:** Automated watershed delineation (707 initial stations), pour point snapping, and derivation of 18 geomorphological features.
3.  **Hydrological Calibration:** Baseflow separation, storm event selection (filtering for quality), and concurrent optimization of $T_c$, $R$, Initial Loss, and Constant Loss using the **Differential Evolution (DE)** algorithm.
4.  **Final Dataset:** Curation resulted in **112 unique watersheds** with 4,248 high-quality parameter sets ($NSE \ge 0.5$).
5.  **ML Modeling:** Training of linear (Elastic Net) and non-linear (Random Forest, SVR) models, followed by rigorous **Leave-One-Out Cross-Validation (LOOCV)** and final hydrologic validation.

## 🗺️ Repository Structure & Reproducibility

For full reproducibility, the notebooks are structured sequentially and should be viewed in the order listed below.

| Step | Notebook Name | Description |
| :--- | :--- | :--- |
| 1 | [Study area](notebooks/preprocessing/1_study_area.ipynb) | Define study region, filter initial gauges, and visualize boundaries. |
| 2 | [DEM tiles downlaod](notebooks/data_donwload/2_DEM_download.ipynb) | Download and assess feasibility of 1/3 arc-second DEM resolution. |
| 3 | [Precipitation data download](notebooks/data_donwload/3_ppt_download.ipynb) | Retrieve historical precipitation data from NOAA FTP. |
| 4 | [Streamflow data download](notebooks/data_donwload/4_streaflow_download.ipynb) | Retrieve historical streamflow data via USGS dataRetrieval. |
| 5 | [Precipitation data preprocessing](notebooks/preprocessing/5_ppt_historical_data_preprocessing.ipynb) | Clean, quality-control, and ensure temporal consistency of precipitation data. |
| 6 | [CRS and projection](notebooks/preprocessing/6_define_utm_zone_projection.ipynb) | Define the Coordinate Reference System strategy for the Ohio Region. |
| 7 | [DEM preprocessing](notebooks/preprocessing/7_dem_preprocessing.ipynb) | Verify DEM tiles, fill pits/depressions, and create Virtual Raster Tiles (VRTs). |
| 8 | [Watershed delineation](notebooks/preprocessing/8_waterhsed_delineation.ipynb) | Delineate watersheds for 707 gauges using D8 algorithm and snap pour points. |
| 9 | [Geomorphological characteristics derivation](notebooks/preprocessing/9_geomorphological_characteristics.ipynb) | Compute 18 static features (Area, Slopes, Ratios, Density, etc.). |
| 10 | [Voronoi polygons - spatial distribution of rainfall](notebooks/preprocessing/10_voronoi_polygons.ipynb) | Create Voronoi polygons for rainfall assignment. |
| 11 | [Line up streamflow and precipitation data](notebooks/preprocessing/11_tabular_data_preprocessing.ipynb) | Align streamflow and precipitation time series data. |
| 12 | [Storm events selection](notebooks/hydrological_model/12_storm_event_selection.ipynb) | Implement filtering criteria to select 20,201 suitable storm events. |
| 13 | [Hydrological parameters optmization](notebooks/hydrological_model/13_ModClark_params_optimization.ipynb) | Calibrate $T_c$, $R$, and loss parameters using Differential Evolution. |
| 14 | [Opmized event exploratory analysis](notebooks/hydrological_model/14_explorartory_analysis.ipynb) | Diagnostic analysis on optimized parameters (e.g., peak flow failure rate). |
| 15 | [Explortaroy data analysis on features](notebooks/ML_model/15_explorartory_data_analysis.ipynb) | EDA on geomorphological features and target parameters ($T_c$, $R$). |
| 16 | [Machine learning model and evaluation](notebooks/ML_model/16_ML_modeling_and_evaluation.ipynb) | Train ML models, apply transformation/scaling, and evaluate performance via $R^2$ and hydrologic simulation (NSE). |

## 🔗 License

This project is licensed under the MIT License.