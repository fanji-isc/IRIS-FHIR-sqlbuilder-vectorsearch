# IRIS FHIR SQL Builder + Vector Search
This repository demonstrates a Proof of Concept (PoC) application leveraging **InterSystems IRIS For Health** to perform semantic search on FHIR data. It combines the capabilities of the **FHIR SQL Builder** for relational data projection and **Vector Search** for meaning-based retrieval.

## Overview
This project enables **semantic search on FHIR data** using InterSystems IRIS. By combining the **FHIR SQL Builder** (for relational projection) with **Vector Search** (for meaning-based retrieval), it allows users to query clinical records using natural language, effectively bridging the gap between complex FHIR structures and human-readable search.

## Features
* **FHIR Repository Integration**: Automated setup and loading of sample FHIR datasets.
* **SQL Projection**: Usage of InterSystems FHIR SQL Builder to expose FHIR data as relational tables without moving data.
* **Vector Database**: Utilization of InterSystems IRIS to store and query high-dimensional embeddings.
* **Semantic Search**: Find clinical records based on meaning (e.g., "heart issues" matches "cardiac arrest").
* **Containerized**: Fully Dockerized setup for easy deployment.

## Getting Started

### Installation
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/fanji-isc/IRIS-FHIR-sqlbuilder-vectorsearch.git](https://github.com/fanji-isc/IRIS-FHIR-sqlbuilder-vectorsearch.git)
    cd IRIS-FHIR-sqlbuilder-vectorsearch
    ```
2.  **Build and Start IRIS For Health**
    ```bash
    docker-compose up -d --build iris4health
    ``` 
3.  **Configure FHIR SQL Builder**
    * **IRIS Portal**: `http://localhost:8080/csp/sys/UtilHome.csp`
        * **Username**: _SYSTEM
        * **Password**: ISCDEMO

    In IRIS Portal, navigate to Health--> FHIR Server Management-->FHIR SQL Builder(left side menu), fill out each section with below information.
    * **Analysis**
        * **Name:FHIRSERVER**
        * **Host:localhost**
        * **Port:52773**
        * **Credentials:New**
            * **Name:Admin**
            * **Username:_SYSTEM**
            * **Password:ISCDEMO**
        * **FHIR Repository URL: /csp/healthshare/demo/fhir/r4**

        In the New FHIR Analysis
        * **FHIR Repository: FHIRSERVER**
        * **Selectivity Percentage:100**
    
    * **Transformation Specifications**
        * Click **Import** and select the `transform1.json` file located in the `fhirdata/` folder of this repository.

    * **Projections**
        * **FHIR Repository:FHIRSERVER**
        * **Transformation Specification:transform1**
        * **Package Name:sql1**

4.  **Accessing FHIR Data via SQL**

    In the Management Portal SQL interface, select DEMO as Namespace you will see the SQL table generated from your QuestionnaireResponse data, ready for querying and analysis.
    ```sql
    SELECT * FROM sql1.QuestionnaireResponse
    ```
    ![sql1query](./sql1query.png)

5.  **Application Setup & Vectorization**

    Now that the data is projected to SQL, start the Python application to generate embeddings and perform searches.

    After running 

    ```bash
    docker-compose up -d --build flask
    ``` 

    you should be able to access your application's user interface at:

    `http://localhost:8000`

6. **Useage & Demo**

    Vector Search
    > **Example: Searching for "diabetes medication"**
    > Searching for **"diabetes medication"** returns not only patients taking drugs like `metformin`, `insulin glargine`, or `acarbose`, but also those whose family history includes **Type 1** or **Type 2 diabetes**. This demonstrates how IRIS Vector Search connects related medical concepts across different fields, such as medications and family history.

    RAG
    > **Example: Multi-Condition Patient Query**
    > When a user asks, **“Which patients have a family history of heart disease and are currently prescribed cholesterol-lowering medication?”**, the system semantically retrieves relevant data from family history and medication fields — identifying links between terms like **cardiac conditions**, **heart failure**, or **coronary disease** with drugs such as `propranolol`, `spironolactone`, or `atorvastatin`.
    > 
    > The LLM then summarizes these findings in natural language, giving clinicians a quick overview of at-risk patients without manual filtering or query writing.

