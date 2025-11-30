# Churn Prediction API

A FastAPI-based machine learning service for customer churn prediction using scikit-learn models.

## 🎯 Project Overview

This service predicts customer churn based on usage patterns, support interactions, and account characteristics. It provides RESTful APIs for training models, making predictions, and monitoring model performance.

## 📊 Dataset Format

The service uses the `churn_dataset.csv` file with the following structure:

### Features:
- `monthly_fee` (float): Monthly subscription fee
- `usage_hours` (float): Hours of service usage in last month
- `support_requests` (int): Number of support requests
- `account_age_months` (int): Account age in months
- `failed_payments` (int): Number of failed payments
- `region` (str): Customer region (europe, asia, america, africa)
- `device_type` (str): Primary device type (mobile, desktop, tablet)
- `payment_method` (str): Payment method (card, paypal, crypto)
- `autopay_enabled` (int): Autopay enabled (0 or 1)

### Target:
- `churn` (int): Churn label (1 if customer churned, 0 if stayed)

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc
   - Health check: http://localhost:8000/health

### Docker Deployment

1. **Using Docker Compose (Recommended):**
   ```bash
   docker-compose up -d
   ```

2. **Using Docker directly:**
   ```bash
   docker build -t churn-api .
   docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api
   ```

## 📡 API Endpoints

### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health and model status

### Model Management
- `POST /model/train` - Train a new churn model
- `GET /model/status` - Get current model status
- `POST /model/load` - Load a previously trained model

### Predictions
- `POST /predict` - Predict churn for a single customer
- `POST /predict/batch` - Predict churn for multiple customers
- `POST /predict/example` - Example prediction for testing

### Data Management
- `POST /data/upload` - Upload and validate dataset
- `GET /data/stats` - Get dataset statistics

## 🔧 Usage Examples

### 1. Train the Model
```bash
curl -X POST "http://localhost:8000/model/train" \
  -H "Content-Type: application/json"
```

### 2. Single Customer Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 29.99,
    "usage_hours": 45.5,
    "support_requests": 2,
    "account_age_months": 12,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1
  }'
```

### 3. Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "monthly_fee": 29.99,
        "usage_hours": 45.5,
        "support_requests": 2,
        "account_age_months": 12,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1
      },
      {
        "monthly_fee": 19.99,
        "usage_hours": 25.0,
        "support_requests": 5,
        "account_age_months": 6,
        "failed_payments": 2,
        "region": "asia",
        "device_type": "desktop",
        "payment_method": "paypal",
        "autopay_enabled": 0
      }
    ]
  }'
```

### 4. Check Model Status
```bash
curl -X GET "http://localhost:8000/model/status"
```

## 🧪 Testing

Run the test suite:
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v
```

## 📁 Project Structure

```
churn-prediction-api/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── churn_dataset.csv      # Sample dataset
├── models/                # Trained model storage
├── api/                   # API related modules
├── ml/                    # Machine learning modules
│   └── churn_model.py     # Churn prediction pipeline
├── schemas/               # Pydantic schemas
│   └── churn.py          # Data models and response schemas
├── core/                  # Core utilities
│   └── config.py         # Configuration settings
└── tests/                 # Test modules
    └── test_main.py      # API tests
```

## 🤖 Model Details

### Algorithm
- **Model Type**: Random Forest Classifier
- **Features**: 9 customer attributes
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Cross-validation**: 80/20 train-test split

### Performance Metrics
The model provides the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

### Risk Levels
Predictions include risk classifications:
- **Low** (< 30% churn probability)
- **Medium** (30-70% churn probability)
- **High** (> 70% churn probability)

## 🔍 Monitoring

### Health Monitoring
The service includes comprehensive health monitoring:
- Model training status
- System metrics (memory, uptime)
- API response times

### Logging
Detailed logging is available in:
- Application logs (structured logging)
- Model training logs
- Error tracking

## ⚙️ Configuration

Environment variables:
- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: info)
- `PORT`: Service port (default: 8000)
- `MODEL_PATH`: Model file path (default: models/churn_model.pkl)
- `DATASET_PATH`: Dataset file path (default: churn_dataset.csv)

## 🛡️ Security Features

- Input validation with Pydantic
- Error handling and sanitization
- CORS middleware
- Rate limiting ready architecture
- Docker security best practices

## 📈 Performance

- Fast async API with FastAPI
- Efficient model serving with joblib
- Batch processing support
- Memory-optimized predictions

## 🚀 Deployment

### Production Deployment
1. **Environment Setup:**
   - Set appropriate environment variables
   - Configure reverse proxy (nginx)
   - Set up monitoring and logging

2. **Scaling:**
   - Use Docker Swarm or Kubernetes
   - Implement load balancing
   - Consider model versioning

3. **Monitoring:**
   - Set up health checks
   - Monitor API performance
   - Track model accuracy over time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational purposes.

## 🆘 Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the health status at `/health`
- Check logs for error details