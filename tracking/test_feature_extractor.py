import numpy as np
import pandas as pd
import pytest


from explainability.feature_extractor import FeatureExtractor;

@pytest.fixture
def frame_shape():
    return (720, 1280, 3)

@pytest.fixture
def mock_tracks():

    return [
        {'track_id': '1',
        'bbox': [np.float64(309.59720742222584), np.float64(232.1016493193435), np.float64(486.14131320727586), 
            np.float64(371.07714489627324)], 
        'class_id': 0, 
        'frame': 2, 
        'image': 'path to jpg'},
        {
            "track_id": 2,
            "bbox": [600, 400, 700, 500],  # x1,y1,x2,y2
            "speed": 12.0
        },
        {
            "track_id": 3,
            "bbox": [300, 450, 360, 520],
            "speed": 8.0
        }
    ]



@pytest.fixture
def timeseries_row(mock_tracks, frame_shape):
    features = FeatureExtractor(
        all_tracks=mock_tracks
    )

    track = mock_tracks[0]
    return features.build_timeseries_row(
        track=track,
        frame_idx=0,
        frame_shape=frame_shape
    )

def test_row_contains_expected_keys(timeseries_row):

    expected_keys = {
        'num_pedestrians',
        'avg_ped_speed',
        'num_vehicles',
        'min_vehicle_distance',
        'max_vehicle_speed',
        'avg_vehicle_speed',
        'vehicle_density'
    }
    assert set(timeseries_row.keys()) == expected_keys

# TEST OTHER METRICS THAT BUILT THE DYNOTEARS framework 


# def test_object_count_correct(timeseries_row):
#     assert timeseries_row["num_objects"] == 2


# def test_mean_speed_correct(mock_tracks, frame_shape):
#     row = build_timeseries_row(mock_tracks, 0, frame_shape)
#     assert row["mean_speed"] == pytest.approx(10.0)


# def test_vehicle_density_correct(mock_tracks, frame_shape):
#     H, W = frame_shape[:2]
#     expected_density = 2 / (H * W)

#     row = build_timeseries_row(mock_tracks, 0, frame_shape)
#     assert row["vehicle_density"] == pytest.approx(expected_density)


# def test_distance_positive(mock_tracks, frame_shape):
#     row = build_timeseries_row(mock_tracks, 0, frame_shape)
#     assert row["mean_distance"] > 0


# # -----------------------------
# # Edge cases
# # -----------------------------

# def test_empty_tracks(frame_shape):
#     row = build_timeseries_row([], frame_idx=5, frame_shape=frame_shape)

#     assert row["frame"] == 5
#     assert row["num_objects"] == 0
#     assert row["mean_speed"] == 0.0
#     assert row["mean_distance"] == 0.0
#     assert row["vehicle_density"] == 0.0


# # -----------------------------
# # Time series tests
# # -----------------------------

# def test_timeseries_dataframe_shape(frame_shape, mock_tracks):
#     tracking_results = [
#         mock_tracks,
#         mock_tracks,
#         []
#     ]

#     frame_shapes = [frame_shape] * 3

#     df = build_timeseries(tracking_results, frame_shapes)

#     assert isinstance(df, pd.DataFrame)
#     assert df.shape[0] == 3
#     assert df.shape[1] == 5


# def test_timeseries_no_nans(frame_shape, mock_tracks):
#     tracking_results = [
#         mock_tracks,
#         [],
#         mock_tracks
#     ]

#     frame_shapes = [frame_shape] * 3

#     df = build_timeseries(tracking_results, frame_shapes)
#     assert not df.isna().any().any()


# def test_frame_index_monotonic(frame_shape, mock_tracks):
#     tracking_results = [mock_tracks] * 4
#     frame_shapes = [frame_shape] * 4

#     df = build_timeseries(tracking_results, frame_shapes)
#     assert list(df["frame"]) == [0, 1, 2, 3]


# # -----------------------------
# # DynoTears compatibility
# # -----------------------------

# def test_dynotears_matrix_ready(frame_shape, mock_tracks):
#     tracking_results = [mock_tracks] * 5
#     frame_shapes = [frame_shape] * 5

#     df = build_timeseries(tracking_results, frame_shapes)

#     matrix = df.drop(columns=["frame"]).to_numpy()

#     assert matrix.ndim == 2
#     assert matrix.dtype != object
