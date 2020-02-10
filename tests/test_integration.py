
import boto3

import pallas


def test_athena():
    client = boto3.client("athena", region_name=pallas.REGION_NAME)
    response = client.start_query_execution(
        QueryString="SELECT 1",
        ResultConfiguration={"OutputLocation": pallas.OUTPUT_LOCATION},
    )
    execution_id = response["QueryExecutionId"]
    state = "RUNNING"
    while state == "RUNNING":
        result = client.get_query_execution(QueryExecutionId=execution_id)
        print(result)
        state = result["QueryExecution"]['Status']['State']
    assert state == "SUCCEEDED"
