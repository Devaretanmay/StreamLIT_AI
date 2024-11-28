# Setup Instructions

## 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## 2. Install Dependencies
```bash
# Install required packages
pip install vanna>=0.7.5 streamlit python-dotenv psycopg2-binary pandas plotly
```

## 3. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5. Access the Application
Once running, the app will be available at:
- Local URL: http://localhost:8501
- Network URL: http://YOUR_IP:8501

## Troubleshooting
- If you get a module not found error, make sure you're in the activated virtual environment
- Verify all dependencies are installed using `pip list`
- Ensure your `.env` file is in the correct location
- Check if streamlit is properly installed using `streamlit --version`

## To Deactivate Virtual Environment
When you're done, you can deactivate the virtual environment:
```bash
deactivate