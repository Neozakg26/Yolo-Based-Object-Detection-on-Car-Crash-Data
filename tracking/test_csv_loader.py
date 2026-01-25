from explainability.metadata import MetaData

if __name__ == "__main__":
    PATH = "C:/Users/neokg/Coding_Projects/yolo-detector/car_crash_dataset/CCD_images/Crash_Table.csv"

    df = MetaData(PATH,scene_id="000001")

    print(f"{df.metadata}") 

