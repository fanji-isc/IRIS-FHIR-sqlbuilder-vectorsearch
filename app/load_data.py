import iris
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import os

# Connection setup
hostname  = "iris4health"       # or "iris" if inside same Docker network
port      = 1972
namespace = "DEMO"
username  = "_SYSTEM"
password  = "ISCDEMO"

conn_url = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
engine = create_engine(conn_url)


def reset_table():
    with engine.begin() as conn:  
        try:
            conn.execute(text("DROP TABLE Sql1.patient_info"))
        except Exception:
            pass



def get_combined_dataframe(source_table, engine, schema="SQL1"):
    query = f"""
        SELECT 
            Allergies,
            City,
            DOB,
            FamilyHistory,
            Medication,
            Name,
            PostalCode,
            State,
            Street
        FROM {schema}.{source_table}
    """
    df = pd.read_sql(query, con=engine)
    df["CombinedData"] = df.astype(str).agg(" | ".join, axis=1)
    return df


def embed_and_load_patients(
    model_name="all-MiniLM-L6-v2"
    ):
    df = get_combined_dataframe("QuestionnaireResponse", engine, schema="SQL1")

    emb_model = SentenceTransformer(model_name)
    embeddings = emb_model.encode(df['CombinedData'].tolist(), normalize_embeddings=True)
    df["patient_vector"] = embeddings.tolist()


    create_sql = """
        CREATE TABLE IF NOT EXISTS sql1.patient_info (
            Allergies      VARCHAR(1000),
            City           VARCHAR(255),
            DOB            VARCHAR(255),
            FamilyHistory  VARCHAR(2000),
            Medication     VARCHAR(1000),
            Name           VARCHAR(255),
            PostalCode     VARCHAR(255),
            State          VARCHAR(255),
            Street         VARCHAR(1000),
            CombinedData   VARCHAR(10000),
            patient_vector VECTOR(FLOAT, 384)
        )
    """

    with engine.connect() as conn:
        with conn.begin(): 
            conn.execute(text(create_sql))

            for _, row in df.iterrows():
                insert_sql = text("""
                    INSERT INTO sql1.patient_info 
                    (Allergies, City, DOB, FamilyHistory, Medication, Name, PostalCode, State, Street, CombinedData, patient_vector) 
                    VALUES (:Allergies, :City, :DOB, :FamilyHistory, :Medication, :Name,
                :PostalCode, :State, :Street, :CombinedData, TO_VECTOR(:patient_vector))
                """)
                conn.execute(insert_sql, {
                    "Allergies": row["Allergies"],
                    "City": row["City"],
                    "DOB": row["DOB"],
                    "FamilyHistory": row["FamilyHistory"],
                    "Medication": row["Medication"],
                    "Name": row["Name"],
                    "PostalCode": row["PostalCode"],
                    "State": row["State"],
                    "Street": row["Street"],
                    "CombinedData": row["CombinedData"],      
                    "patient_vector": str(row["patient_vector"])
                })


# if __name__ == "__main__":
#     reset_table()
#     get_combined_dataframe()
#     embed_and_load_patients()
if __name__ == "__main__":
    print("Resetting table…")
    reset_table()
    print("Embedding and loading patients…")
    embed_and_load_patients()
    print("Done.")

  