# Forecasting the Stock Market with Watson Studio

## Summary 
This code pattern demonstrates how subject matter experts and data scientists can leverage IBM Watson Studio to automate data mining and the training of time series forecasters using open-source machine learning libraries, or the built-in graphical tool integrated into Watson Studio. It applies ARIMA algorithms (Autoregressive Integrated Moving Average) and other advanced techniques to construct mathematical models capable of predicting trends based on data from the past.

## Description
Using the IBM Watson Studio and other popular open-source Python libraries for data science, this code pattern provides an example of data science workflow which attempts to predict the end-of-day value of S&P 500 stocks based on historical data. It includes the data mining process, that uses the Quandl API – a marketplace for financial, economic and alternative data delivered in modern formats for today's analysts.

## After completing this code pattern, you’ll understand how to:
*  Use Jupyter Notebooks in Watson Studio to mine financial data using public APIs;
*  Use specialized Watson Studio tools like Data Refinery to prepare data for model training;
*  Build, train, and save a timeseries model from extracted data, using open-source Python libraries and/or the built-in graphical Modeler Flow in Watson Studio;
*  Interact with IBM Cloud Object Storage to store and access mined and modeled data; 
*  Store a model created with Modeler Flow and interact with the Watson Machine Learning service using the Python API; and
*  Generate graphical visualizations of timeseries data using Pandas and Bokeh.
 
## Detailed Instructions

This tutorial will assume you know how to provision services from the Catalog, using the IBM Cloud Web Portal. Three services are required for this code pattern: `IBM Cloud Object Storage`, `Watson Machine Learning` and `Watson Studio`. After you create one instance of each service, you can proceed (The Lite plans are sufficient for running this Code pattern).

### Table of Contents

1. [Creating a New Project at Watson Studio](https://github.com/vanderleipf/ibmdegla-ws-projects/tree/master/forecasting-the-stock-market#1--creating-a-new-project-at-watson-studio)

2. [Mining Data and Making forecasts with a Python Notebook](https://github.com/vanderleipf/ibmdegla-ws-projects/tree/master/forecasting-the-stock-market#2--mining-data-and-making-forecasts-with-a-python-notebook)

i. [Configuring the Quandl API-KEY](https://github.com/vanderleipf/ibmdegla-ws-projects/tree/master/forecasting-the-stock-market#i--configuring-the-quandl-api-key)

ii. [Configuring the IBM Cloud Object Storage credentials in the Notebook](https://github.com/vanderleipf/ibmdegla-ws-projects/tree/master/forecasting-the-stock-market#ii--configuring-the-ibm-cloud-object-storage-credentials-in-the-notebook)

iii. [Importing the Mined Data as an Asset into the Watson Studio Project](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#iii--importing-the-mined-data-as-an-asset-into-the-watson-studio-project)

3. [Cleansing Data with Data Refinery](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#3--cleansing-data-with-data-refinery)

4. [Making Forecasts with SPSS Modeler Flow](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#4--making-forecasts-with-spss-modeler-flow)

5. [Visualizing Modeler Flow Results with a Python Notebook](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#5--visualizing-modeler-flow-results-with-a-python-notebook)

6. [Deploying a Modeler Flow Model in Watson Machine Learning](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#6--deploying-a-modeler-flow-model-in-watson-machine-learning)

7. [Interacting with the Watson Machine Learning API](https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/README.md#7--interacting-with-the-watson-machine-learning-api)

<hr>

### 1.  Creating a New Project at Watson Studio

After creating an instance of Watson Studio, you will see the following screen:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/04.png)

After clicking at the `Get Started` button you will be directed to Watson Studio main page, shown below. You can then click on `Create a project` and then select the `Standard` option.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/05.png)

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/06.png)

Now you will be directed to the project creation page shown below. You must give a name for your project and also select the `IBM Cloud Object Storage` service you provisioned beforehand, to be used as data storage.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/07.png)

<hr>

### 2.  Mining Data and Making forecasts with a Python Notebook

After your project is created, you will be directed to the project overview page. In this page you can oversee some general aspects of your project, such as collaborators and data consumed. You should now go to the `Assets` tab, as shown in the picture below.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/08.png)

At the assets tab, click on the `Add to project` blue button on the top right corner, and select the asset type as `Notebook`:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/09.png)

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/10.png)

Now you will be directed to the notebook creation page. Give a name to the Notebook, and select the desired Python runtime (you can choose the free one). Then, click on the `From URL` tab and paste the following link at the `Notebook URL` field:
`https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/jupyter-notebooks/forecasting-the-stock-market.ipynb`. Alternatively you can choose the `From file` option and upload the `forecasting-the-stock-market.ipynb` file, if you have downloaded this repository to your local computer.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/11.png)

After click on `Create notebook`, Watson Studio will load the file and start the kernel and you will be directed to your notebook environment. From now on the Python Notebook is ready and can be started by clicking at the `RUN` button indicated in the picture below. You can read the instructions and comments in the notebook and start executing cell by cell.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/12.png)

There are only two steps that require further action now - the provisioning of an API-KEY for the Quandl database (that can be done for free <a href="https://www.quandl.com/sign-up-modal?defaultModal=showSignUp">at the Quandl website</a>, and the configuration of the IBM Cloud Object Storage credentials at section 4 of the Notebook.

#### i.  Configuring the Quandl API-KEY

After registering for a free API-KEY at the Quandl website, you just need to write it at the indicated cell, as shown below.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/13.png)

After this step you can execute all the cells - where all the data science is done! - until section 4

#### ii.  Configuring the IBM Cloud Object Storage credentials in the Notebook

This step is required so you can export the mined data and also the results of the forecaster to IBM Cloud Object Storage. Using the IBM COS API you can then use the stored data as you wish (publication, further analysis with different tools, etc). In the cell indicated at the picture below, you must complete the `variable cos_credentials` with your IBM COS credentials.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/14.png)

There is an easy way to do this. First, click at the indicated button in the top right corner of the screen and upload the `AAPL.csv` file (provided in this repository, at the `data-samples` directory).

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/15.png)

The file will appear at the right side panel. Click at `Insert to code` and then `Insert credentials`, as shown below. Your credentials will appear at the selected cell.

#### iii.  Importing the Mined Data as an Asset into the Watson Studio Project

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/16.png)

Don't forget that the variable with the credentials must be named `cos_credentials` for the defined function (in the next cell) to work. You are now ready to upload the two csv files generated by the analysis to the IBM Cloud Object Storage service.

After executing all the remaining cells in the Notebook, if you go back to the `Assets` tab and click in the indicated buttons below, you will be able to see some new files - in the picture: `AAPL.csv` (the file you manually uploaded), `IBM.csv` (IBM financial data downloaded from Quandl) and `IBM_future.csv` (the predictions generated by the machine learning model). Import these files as assets to your project.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/17.png)

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/18.png)

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/19.png)

I'll be able to see the new data assets at the `Assets` tab:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/20.png)

<hr>

### 3.  Cleansing Data with Data Refinery

In this step we are going to use Data Refinery to cleanse data - the imported csv files (AAPL.csv, or other financial data collected by you with the Python Notebook). First, click at the `Add to project` blue button at the top right corner and select a new `DATA REFINERY FLOW`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/21.png)

Next, you should choose the target csv file (the `AAPL.csv` file is chosen in this example).

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/22.png)

After Data Refinery reads the target file, you will see the following screen:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/23.png)

From the sample data shown at the table in the picture above, we can see that there are some problems with the source data: the columns are unnamed, the data types are incorrect, there is an useless index column, and the first row of the table (the labels of the columns) should be dropped. In the next following steps we are going to create a Flow of actions to fix these problems.

First we remove the first row (labels) by clicking at the "triple dots" at `COLUMN1` and then `Remove empty rows`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/24.png)

Then we remove the first column (the useless one) by clicking at the triple dots at `COLUMN1` and then `Remove`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/25.png)

Now we'll change the data types. First, click at the triple dots at `COLUMN2` followed by `CONVERT COLUMN` and then choose `Date`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/26.png)

In the next screen (shown below), choose the `ymd` order (year-month-day) at the left side panel and click `Apply`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/27.png)

Lastly, you should change the data types for `COLUMN3`, `COLUMN4`, `COLUMN5`, and `COLUMN6` from String to Decimal. This can be done by clicking at the triple dots followed by `CONVERT COLUMN` and then choosing `Decimal`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/28.png)

After converting the four columns to Decimal types, you should see something like this (Five columns (one with type Date and four with type Decimal) and a flow with 7 steps): 

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/29.png)

If everything is correct, click at the `Run Data Refinery Flow` button at the top right corner (shown in the picture above), and then click at `Save and Run Flow` in the next screen.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/30.png)

After the Data Refinery Flow is completed, you will be able to see a new csv file at the Assets tab (`AAPL.csv_shaped.csv`):

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/32.png)

This csv file will be the input for SPSS Modeler Flow that will be created next.

<hr>

### 4.  Making forecasts with SPSS Modeler Flow

With the cleansed `AAPL.shaped_csv.csv` file we can proceed to create the Modeler Flow for forecasting future stock values. Click at the `Add to project` blue button at the top right corner and select a new `MODELER FLOW`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/33.png)

In the Modeler Flow creation page, select the `From file` option and upload the `forecasting-stocks-with-spss-modeler.str` file (provided in this repository). Click at `Create`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/34.png)

After Watson Studio finishes loading, you will see the flow shown in the picture below. This flow consists on an `Data Asset` block, where we set the source file; a `Filter` block that is used to rename the columns; a `Type` block, used to set the target and input columns; a `Sample` block, to split the source data into train and test datasets; and a `Time Series` modeler block, to generate the predictions. The dark blue blocks are outputs.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/35.png)

Before executing it, we need to set the source data. To do this, click at the `Data Asset` block and select in the right panel the `<STOCK_TICKER>.shaped_csv.csv` file (AAPL in this example).

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/36.png)

Click `Save` and then `Run` at the indicated button:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/37.png)

After the flow finishes execution, you will be able to see the outputs at the right panel.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/38.png)

Clicking at the `Multivar Plot` block we can see the graph with historical and predicted data

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/39.png)

This plot don't have interactive capabilities like Bokeh, that we previously used in a Python Notebook. In the next section it is presented a simple way to visualize this data generated by the modeler flow in a Python notebook.

<hr>

### 5.  Visualizing Modeler Flow Results with a Python Notebook

Just as it was done before, add a new Notebook asset to your project:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/40.png)

Select the `From URL` option, and paste the following link at the indicated field: 
`https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/jupyter-notebooks/visualizing-spss-modeler-flow-results.ipynb`. This is the secondary notebook provided in this repository.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/41.png)

After Watson Studio finishes loading the Python kernel, you have - again - to configure the IBM Cloud Object Storage service credentials in one of the cells of the Notebook. You can use the same easy procedure presented earlier to do this.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/42.png)

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/43.png)

After configuring IBM COS, the Python Notebook will be able to fetch the csv file with the modeler flow results. This data is then loaded into a Pandas DataFrame for plotting with the Bokeh package.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/44.png)

Now you are able to interact with the plot and better analyze the results.

Feel free to change the machine learning models and modeler flow parameters. There are also thousands of several other stocks available at the Quandl database for further analysis.

<hr>

### 6.  Deploying a Modeler Flow Model in Watson Machine Learning

In this section you'll learn on how to store a model trained with Watson Studio Modeler Flow and also how to make API calls to your stored model, deployed as a Web Service in Watson Machine Learning.

First, go back to the Modeler Flow canvas and right-click the `Table` output node and select `Save branch as a model`, as indicated in the picture below:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/45.png)

You'll be directed to the `Save Model` page. Save the model as `Scoring Branch` and give a name and description to your model. In this case, the model predicts the closing end-of-day value for Apple Inc. stocks. You also need to select an instance of Watson Machine Learning, previously created in the beggining of this tutorial. In the picture below the WML instance is named `ibmdegla-watson-ml`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/46.png)

After saving the model you'll be able to see it in the Watson Studio project `Assets` tab. Click on the saved model (in the picture below the model is named `Closing-Value-AAPL-Forecaster`):

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/47.png)

You will see now some information about your saved model, like the input schema and running environment. Click on the deployments tab.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/48.png)

Click in `Add deployment`, in the right side of the screen:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/49.png)

Give a name and description to your model deployment and click `Save`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/50.png)

After the deployment is finished, you will see `DEPLOY_SUCCESS` in the status field. Click in the deployment (in the picture below the deployment is named `My-Deployment`).

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/51.png)

Then click at the `Implementation` tab

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/52.png)

Copy the `Scoring End-Point` link, as it will be needed later when calling the Watson Machine Learning API.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/53.png)

<hr>

### 7.  Interacting with the Watson Machine Learning API

After successfully deploying the Apple Inc. stock value forecaster in a Watson Machine Learning instance, you are now able to send new input data to be scored by the model using the generated API. In this section it's demonstrated how to interact with Watson Machine Learning using Python.

Just as it was done before, add a new Notebook asset to your project:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/40.png)

Select the `From URL` option, and paste the following link at the indicated field: 
`https://github.com/vanderleipf/ibmdegla-ws-projects/blob/master/forecasting-the-stock-market/jupyter-notebooks/using-watson-machine-learning.ipynb`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/54.png) 

After Watson Studio finishes loading the Python kernel, you can execute cell by cell until the part where IBM COS must be configured.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/55.png)

You have - again - to configure the IBM Cloud Object Storage service credentials. You can use the same easy procedure presented earlier to do this.
 
![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/56.png)

After configuring IBM COS, the Python Notebook will be able to fetch the csv file with the mined Apple Inc stock data.

You then execute the next cells until section `3.2: Setting Up WML Credentials`.

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/57.png)

Paste the `Scoring End-Point` link you copied before into the `deployment_endpoint` variable, and your Watson Machine Learning credentials into `wml_credentials` variable. The WML credentials can be found inside the page of the service instance in the IBM Cloud web portal, as shown below:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/58.png)

After setting these variables, you can then execute all cells in this notebook, until you can see the predicted results using Bokeh at the end:

![alt text](https://raw.githubusercontent.com/vanderleipf/ibmdegla-ws-projects/master/forecasting-the-stock-market/screenshots/59.png)

### Thank you for completing this journey!
