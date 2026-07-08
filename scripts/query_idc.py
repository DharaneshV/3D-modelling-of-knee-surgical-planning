from idc_index import IDCClient
import pandas as pd

client = IDCClient.client()

# Query the local IDC index for CT scans containing knee/extremity/lower limb structures
# that are 3D volumes (high instance count)
query = """
SELECT 
    collection_id,
    PatientID,
    StudyInstanceUID,
    SeriesInstanceUID,
    instanceCount,
    BodyPartExamined,
    SeriesDescription
FROM index
WHERE Modality = 'CT' 
  AND (BodyPartExamined = 'EXTREMITY' OR BodyPartExamined = 'LOWER LIMB' OR BodyPartExamined = 'KNEE')
  AND (SeriesDescription LIKE '%KNEE%' OR SeriesDescription LIKE '%LEG%' OR SeriesDescription LIKE '%THIGH%' OR SeriesDescription LIKE '%FEMUR%' OR SeriesDescription LIKE '%TIBIA%')
  AND instanceCount > 80
LIMIT 50
"""

try:
    results = client.sql_query(query)
    # Convert to DataFrame if it's not already
    if not isinstance(results, pd.DataFrame):
        df = pd.DataFrame(results)
    else:
        df = results
        
    print(f"Found {len(df)} matching series:")
    print(df.to_string())
except Exception as e:
    print(f"Error querying IDC: {e}")
