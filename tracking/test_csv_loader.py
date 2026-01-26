from explainability.metadata import MetaData

if __name__ == "__main__":
    PATH = "/datasets/nmaja/Crash_Table.csv"

    df = MetaData(PATH,scene_id="000001")

    print(f"{df.metadata}") 

