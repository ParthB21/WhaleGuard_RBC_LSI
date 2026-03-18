# WhaleGuard_RBC_LSI
Official repository for Project WhaleGuard by team Neural Network Navigators for RBC Borealis Let's SOLVE It Undergraduate Mentorship Program, Spring 2026 cohort. We are building a predictive platform to forecast North Atlantic Right Whale movements ~72 hours in advance.

## The Problem
Current solutions are primarily reactive, utilizing acoustic buoys or satellite detection to flag whales only after they have entered a shipping lane. These systems often force vessels to brake suddenly, which disrupts supply chains. Furthermore, enterprise-grade systems are often too expensive for smaller vessels, such as fishing and lobster boats, leaving them without AI-enabled protection.

## Our Solution
WhaleGuard shifts maritime safety from a reactive model to a predictive forecasting system. The platform aims to visualize whale movements approximately 72 hours in advance to prevent collisions before they occur.

## Data and Validation
The project utilizes three main categories of open-source data to train and validate the predictive models:
* **Training Data**: Includes whale coordinates from the NOAA Right Whale Sighting Advisory System (2017-2025), NASA ocean environment data (Chlorophyll-a and Sea Surface Temperature), and acoustic data from CIOOS Atlantic and the NOAA NCEI Passive Acoustic Data Archive.
* **Validation**: The team uses the NOAA Unusual Mortality Event dataset to perform "Hindcast Validation," reconstructing ocean conditions from past fatal accidents to see if the model would have correctly flagged those zones as high-risk.
* **Traffic Density**: Real-time shipping lane density from the MyShipTracking API is used to calculate the economic viability of proposed detours.
