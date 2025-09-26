# Model Files for Deployment

This directory should contain the following files for full ML functionality:

## Required Files:
- `model_returns.h5` - Trained LSTM model
- `scaler_X.pkl` - Feature scaler (for input data)
- `scaler_y.pkl` - Target scaler (for predictions)

## File Status:
- ✅ Files are included in the repository
- ✅ Compatible with Render deployment
- ✅ Demo mode available if files are missing

## Notes:
If model files are too large for Git (>100MB), consider using:
1. Git LFS (Large File Storage)
2. External storage (Google Drive, AWS S3)
3. Model compression techniques

## Demo Mode:
The app automatically detects missing model files and switches to demo mode, showing:
- Real-time stock data visualization
- Technical analysis with moving averages
- Historical price charts
- Volume analysis