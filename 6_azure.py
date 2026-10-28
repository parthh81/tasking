"python -m pip install --upgrade pip setuptools wheel"
"python -m pip install azure-ai-ml azure-identity"

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Data, command
from azure.ai.ml.constants import AssetTypes

def run_azure_ml_operations():
    print("--- AZURE MACHINE LEARNING STUDIO SETUP ---")
    
    # =====================================================================
    # IMPORTANT: You must replace these with your actual Azure ML details!
    # If you don't have an Azure account, this script will intentionally fail.
    # =====================================================================
    SUBSCRIPTION_ID = "your-azure-subscription-id"
    RESOURCE_GROUP = "your-resource-group-name"
    WORKSPACE_NAME = "your-aml-workspace-name"

    # 1. AUTHENTICATE AND CONNECT TO AZURE
    print("Authenticating with Azure...")
    try:
        # DefaultAzureCredential handles login automatically if you have the Azure CLI installed
        # and are logged in, or it will prompt you in your browser.
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        print(f"Successfully connected to Workspace: {ml_client.workspace_name}")
    except Exception as e:
        print("\n[ERROR] Could not connect to Azure.")
        print("Did you update the SUBSCRIPTION_ID, RESOURCE_GROUP, and WORKSPACE_NAME variables?")
        print(f"Exact error: {e}")
        return

    # 2. CREATE A COMPUTE CLUSTER (The cloud server that will run your code)
    compute_name = "cpu-cluster-basic"
    print(f"\nChecking for compute cluster: '{compute_name}'...")
    
    try:
        # Check if the cluster already exists
        compute_target = ml_client.compute.get(compute_name)
        print(f"Compute target '{compute_name}' already exists.")
    except Exception:
        print(f"Creating new compute target '{compute_name}' (this takes a few minutes in the cloud)...")
        # Define the server specs (Standard_DS11_v2 is a basic, cheap virtual machine)
        cluster_config = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_DS11_v2",
            min_instances=0, # Scales down to 0 when not in use to save money
            max_instances=2,
            idle_time_before_scale_down=120
        )
        compute_target = ml_client.compute.begin_create_or_update(cluster_config).result()
        print("Compute target created successfully!")

    # 3. REGISTER A DATASET (Data Asset) IN AZURE
    # We will use the same diabetes dataset you used in Task 3, but store it in Azure
    print("\nRegistering Data Asset in Azure ML Studio...")
    my_data = Data(
        path="https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        type=AssetTypes.URI_FILE,
        description="Pima Indians Diabetes dataset directly from a web URL",
        name="diabetes_data_basic",
        version="1"
    )
    ml_client.data.create_or_update(my_data)
    print("Data Asset 'diabetes_data_basic' registered successfully!")

    # 4. CONFIGURE A MACHINE LEARNING JOB
    print("\nConfiguring the ML training job...")
    # This defines the command Azure will run on the remote server. 
    # Usually, you point 'code' to a folder containing a script like your Task 3 code.
    job = command(
        code=".",  # Uploads the current local folder to Azure
        command="echo 'Azure ML Job Started!' && echo 'If we had a training script here, Azure would run it now on the diabetes data: ${{inputs.diabetes}}'",
        inputs={
            "diabetes": ml_client.data.get(name="diabetes_data_basic", version="1").id
        },
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest", # Pre-built Microsoft environment
        compute=compute_name,
        display_name="my-first-azure-job",
        experiment_name="basic-ml-operations"
    )

    # 5. SUBMIT THE JOB TO THE CLOUD
    print("\nSubmitting the job to Azure ML Studio...")
    returned_job = ml_client.jobs.create_or_update(job)
    print("\n" + "=" * 50)
    print("Job submitted successfully!")
    print(f"Job Name: {returned_job.name}")
    print(f"Check your Azure ML Studio dashboard to see it running!")
    print("=" * 50)

if __name__ == "__main__":
    run_azure_ml_operations()