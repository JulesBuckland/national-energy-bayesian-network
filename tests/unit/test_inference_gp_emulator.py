import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.inference.gp_emulator import (
    load_data, train_gp, plot_validation, save_model, validate_saved, main,
    evaluate_gp, check_acceptance, prepare_train_test_data,
)
import src.inference.gp_emulator as gp_emulator

@patch('src.inference.gp_emulator.COMBINED_CSV')
@patch('src.inference.gp_emulator.pd.read_csv')
def test_load_data(mock_read_csv, mock_combined):
    mock_combined.exists.return_value = True
    df = pd.DataFrame({
        'status': ['ok', 'error', 'ok'],
        'floor_area': [1, 2, 3],
        'wall_u': [1, 2, 3],
        'ach': [1, 2, 3],
        'wwr': [1, 2, 3],
        'form_code': [1, 2, 3],
        'hdd': [1, 2, 3],
        'T_h': [100, 200, -10]
    })
    mock_read_csv.return_value = df
    res = load_data()
    assert len(res) == 1
    assert res.iloc[0]['status'] == 'ok'
    assert res.iloc[0]['T_h'] == 100

@patch('src.inference.gp_emulator.COMBINED_CSV')
def test_load_data_missing(mock_combined):
    mock_combined.exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_data()

@patch('src.inference.gp_emulator.GaussianProcessRegressor')
@patch('src.inference.gp_emulator.STATS_PATH')
@patch('builtins.open')
def test_train_gp(mock_open, mock_stats_path, mock_gpr):
    # Create mock GP
    mock_gp = MagicMock()
    mock_gp.predict.side_effect = lambda X, return_std: (np.random.rand(len(X)), np.random.rand(len(X)))
    mock_gp.kernel_ = "mock_kernel"
    mock_gpr.return_value = mock_gp
    
    df = pd.DataFrame({
        'floor_area': np.random.rand(600),
        'wall_u': np.random.rand(600),
        'ach': np.random.rand(600),
        'wwr': np.random.rand(600),
        'form_code': np.random.rand(600),
        'hdd': np.random.rand(600),
        'T_h': np.random.rand(600)
    })
    
    with patch('src.inference.gp_emulator.r2_score', return_value=0.995):
        gp, scaler, X_test, y_test, y_pred, y_std, stats = train_gp(df)
        
    assert gp == mock_gp
    assert stats['acceptance_met'] is True

@patch('src.inference.gp_emulator.plt')
def test_plot_validation(mock_plt):
    mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
    y_test = np.array([1, 2])
    y_pred = np.array([1.1, 1.9])
    y_std = np.array([0.1, 0.1])
    plot_validation(y_test, y_pred, y_std, 0.99)
    mock_plt.savefig.assert_called_once()

@patch('src.inference.gp_emulator.joblib.dump')
def test_save_model(mock_dump):
    save_model("gp", "scaler")
    mock_dump.assert_called_once()

@patch('src.inference.gp_emulator.MODEL_PATH')
@patch('src.inference.gp_emulator.STATS_PATH')
@patch('src.inference.gp_emulator.joblib.load')
@patch('src.inference.gp_emulator.load_data')
@patch('builtins.open')
def test_validate_saved(mock_open, mock_load_data, mock_load, mock_stats_path, mock_model_path):
    mock_model_path.exists.return_value = True
    mock_stats_path.exists.return_value = True
    
    mock_gp = MagicMock()
    mock_gp.predict.return_value = (np.array([1.0]), np.array([0.1]))
    mock_scaler = MagicMock()
    mock_load.return_value = {"gp": mock_gp, "scaler": mock_scaler}
    
    mock_df = pd.DataFrame({
        'floor_area': [1], 'wall_u': [1], 'ach': [1], 'wwr': [1], 'form_code': [1], 'hdd': [1], 'T_h': [100]
    })
    mock_load_data.return_value = mock_df
    
    with patch('src.inference.gp_emulator.json.load', return_value={"r2": 0.99}):
        validate_saved()

@patch('src.inference.gp_emulator.MODEL_PATH')
def test_validate_saved_missing_model_raises(mock_model_path):
    """A missing saved model must raise, not silently return: the pipeline
    audit flagged this exact silent-skip pattern as a way for a broken
    validation stage to vanish with no error."""
    mock_model_path.exists.return_value = False
    with pytest.raises(FileNotFoundError, match="No saved model"):
        validate_saved()


def test_evaluate_gp_matches_hand_computed_metrics():
    """y_test=[10,20,30], y_pred=[12,18,33].
    residuals = pred-test = [2,-2,3]
    MAE  = mean(|2|,|2|,|3|) = 7/3 = 2.3333...
    RMSE = sqrt(mean(4,4,9)) = sqrt(17/3) = sqrt(5.6667) = 2.38048...
    R2   = 1 - SS_res/SS_tot
           SS_res = 4+4+9 = 17
           mean(y_test) = 20; SS_tot = (10-20)^2+(20-20)^2+(30-20)^2 = 100+0+100 = 200
           R2 = 1 - 17/200 = 0.915
    """
    y_test = np.array([10.0, 20.0, 30.0])

    class FakeGP:
        def predict(self, X, return_std=True):
            return np.array([12.0, 18.0, 33.0]), np.array([0.5, 0.5, 0.5])

    metrics = evaluate_gp(FakeGP(), X_test_s=np.zeros((3, 1)), y_test=y_test)

    assert metrics["r2"] == pytest.approx(0.915, abs=1e-9)
    assert metrics["mae"] == pytest.approx(7 / 3, abs=1e-9)
    assert metrics["rmse"] == pytest.approx((17 / 3) ** 0.5, abs=1e-9)
    np.testing.assert_array_equal(metrics["y_pred"], [12.0, 18.0, 33.0])
    np.testing.assert_array_equal(metrics["y_std"], [0.5, 0.5, 0.5])


def test_prepare_train_test_data_uses_all_rows_by_default():
    df = pd.DataFrame({column: np.arange(20, dtype=float) for column in gp_emulator.FEATURES})
    df[gp_emulator.TARGET] = np.arange(20, dtype=float) + 1
    X_train, X_test, _, y_train, y_test, _ = prepare_train_test_data(df)
    assert len(y_train) + len(y_test) == len(df)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_prepare_train_test_data_subsampling_is_explicit():
    df = pd.DataFrame({column: np.arange(20, dtype=float) for column in gp_emulator.FEATURES})
    df[gp_emulator.TARGET] = np.arange(20, dtype=float) + 1
    _, _, _, y_train, _, _ = prepare_train_test_data(df, max_train_points=5)
    assert len(y_train) == 5


def test_check_acceptance_passes_above_threshold():
    check_acceptance(0.995, threshold=0.99)  # must not raise


def test_check_acceptance_raises_below_threshold():
    # Threshold passed explicitly so this doesn't depend on TEST_MODE's
    # environment-dependent default (see settings.GP_ACCEPTANCE_R2).
    with pytest.raises(ValueError, match=r"R² = 0.9000 < 0.99"):
        check_acceptance(0.90, threshold=0.99)


@patch('src.inference.gp_emulator.validate_saved')
def test_main_validate(mock_validate):
    main(validate=True)
    mock_validate.assert_called_once()

@patch('src.inference.gp_emulator.load_data')
@patch('src.inference.gp_emulator.train_gp')
@patch('src.inference.gp_emulator.save_model')
@patch('src.inference.gp_emulator.plot_validation')
def test_main(mock_plot, mock_save, mock_train, mock_load):
    mock_train.return_value = ("gp", "scaler", "X_test", "y_test", "y_pred", "y_std", {"r2": 0.99})
    main(validate=False)
    mock_load.assert_called_once()
    mock_train.assert_called_once()
    mock_save.assert_called_once()
    mock_plot.assert_called_once()
