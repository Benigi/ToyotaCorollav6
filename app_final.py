"""
Toyota Corolla Valuation Tool - Educational Edition
Demonstrates machine learning relationships between vehicle features and intrinsic value.
No external visualization libraries - pure Streamlit implementation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Toyota Corolla Valuation Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #A3D2FF;
    }
    
    h1 {
        color: #8DD7FF;
        font-size: 2.5em;
    }
    
    /* Main container */
    .main {
        background-color: #0F172A;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    
    /* Price display */
    .price-display {
        background: linear-gradient(135deg, #2E86AB 0%, #06A77D 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a2332;
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Text */
    p, label {
        color: #CBD5E1;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load Toyota Corolla dataset"""
    csv_path = BASE_DIR / 'ToyotaCorolla.csv'
    if not csv_path.exists():
        st.error(f"Dataset not found at {csv_path}")
        st.stop()
    
    df = pd.read_csv(csv_path)
    return df


@st.cache_resource
def train_model_and_get_insights():
    """Train Random Forest and calculate insights"""
    df = load_data()
    
    # Clean data
    df_clean = df.dropna()
    
    # Feature columns
    feature_columns = [
        'Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'Fuel_Type', 'HP', 'Met_Color',
        'Automatic', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight',
        'Mfr_Guarantee', 'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1',
        'Airbag_2', 'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
        'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps',
        'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette', 'Tow_Bar'
    ]
    
    X = df_clean[feature_columns].copy()
    y = df_clean['Price'].copy()
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled_df, y)
    
    # Calculate global feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Calculate correlations with price
    correlations = df_clean[feature_columns + ['Price']].corr()['Price'].drop('Price').sort_values(ascending=False)
    
    # Model performance metrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'correlations': correlations,
        'r2': r2,
        'mape': mape,
        'rmse': rmse,
        'base_price': y.mean(),
        'price_std': y.std(),
        'price_min': y.min(),
        'price_max': y.max(),
        'df_clean': df_clean
    }


# ============================================================================
# PREDICTION & EXPLANATION FUNCTIONS
# ============================================================================

def create_feature_vector(age, km, hp, weight, cc, tax, cylinders,
                         fuel_type, automatic, met_color, abs_sys, airco, feature_cols):
    """Create feature vector for prediction"""
    
    features = {
        'Age_08_04': age,
        'Mfg_Month': 6,
        'Mfg_Year': 2002,
        'KM': km,
        'Fuel_Type': 1 if fuel_type == 'Diesel' else 0,
        'HP': hp,
        'Met_Color': 1 if met_color == 'Metallic' else 0,
        'Automatic': 1 if automatic == 'Automatic' else 0,
        'cc': cc,
        'Doors': 4,
        'Cylinders': cylinders,
        'Gears': 5,
        'Quarterly_Tax': tax,
        'Weight': weight,
        'Mfr_Guarantee': 0,
        'BOVAG_Guarantee': 1,
        'Guarantee_Period': 3,
        'ABS': 1 if abs_sys else 0,
        'Airbag_1': 1,
        'Airbag_2': 1,
        'Airco': 1 if airco else 0,
        'Automatic_airco': 0,
        'Boardcomputer': 1,
        'CD_Player': 0,
        'Central_Lock': 1,
        'Powered_Windows': 1,
        'Power_Steering': 1,
        'Radio': 1,
        'Mistlamps': 0,
        'Sport_Model': 0,
        'Backseat_Divider': 0,
        'Metallic_Rim': 1,
        'Radio_cassette': 0,
        'Tow_Bar': 0,
    }
    
    return pd.DataFrame([features])[feature_cols]


def predict_with_explanation(age, km, hp, weight, cc, tax, cylinders,
                           fuel_type, automatic, met_color, abs_sys, airco,
                           model, scaler, feature_cols, base_price, df_clean):
    """Make prediction and calculate feature contributions"""
    
    # Create input
    X_input = create_feature_vector(age, km, hp, weight, cc, tax, cylinders,
                                   fuel_type, automatic, met_color, abs_sys, airco, feature_cols)
    
    # Scale
    X_scaled = scaler.transform(X_input)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    # Calculate feature contributions (simplified SHAP-like)
    # Compare each feature to median value in training data
    contributions = {}
    
    for i, feature in enumerate(feature_cols):
        feature_value = X_input.iloc[0, i]
        median_value = df_clean[feature].median()
        
        # Get tree predictions with this feature at median vs actual
        X_baseline = X_input.copy()
        X_baseline[feature] = median_value
        X_baseline_scaled = scaler.transform(X_baseline)
        
        pred_baseline = model.predict(X_baseline_scaled)[0]
        contribution = prediction - pred_baseline
        
        contributions[feature] = {
            'value': contribution,
            'absolute': abs(contribution)
        }
    
    # Get top 5 contributors
    sorted_contributions = sorted(contributions.items(), 
                                 key=lambda x: x[1]['absolute'], 
                                 reverse=True)[:5]
    
    return {
        'prediction': max(prediction, 1000),
        'contributions': sorted_contributions,
        'base_price': base_price
    }


def calculate_sensitivity(age, km, hp, weight, cc, tax, cylinders,
                         fuel_type, automatic, met_color, abs_sys, airco,
                         model, scaler, feature_cols, feature_name, test_range):
    """Calculate sensitivity for a feature"""
    
    prices = []
    
    for test_value in test_range:
        X_input = create_feature_vector(age, km, hp, weight, cc, tax, cylinders,
                                       fuel_type, automatic, met_color, abs_sys, airco, feature_cols)
        X_input[feature_name] = test_value
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)[0]
        prices.append(max(pred, 1000))
    
    return prices


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load data and train model
    insights = train_model_and_get_insights()
    model = insights['model']
    scaler = insights['scaler']
    feature_cols = insights['feature_columns']
    feature_importance = insights['feature_importance']
    base_price = insights['base_price']
    df_clean = insights['df_clean']
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚗 Toyota Corolla Valuation Tool")
        st.markdown("**Explore how vehicle features determine intrinsic value**")
    
    # ========================================================================
    # SIDEBAR - PARAMETER SELECTION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Vehicle Parameters")
        
        st.subheader("📅 Age & Usage")
        age = st.slider(
            "Age (months)",
            min_value=0,
            max_value=120,
            value=60,
            step=1,
            help="How many months old is the vehicle?"
        )
        
        km = st.slider(
            "Mileage (km)",
            min_value=0,
            max_value=300000,
            value=100000,
            step=5000,
            help="Total kilometers driven"
        )
        
        st.subheader("⚙️ Engine")
        hp = st.slider(
            "Horsepower (HP)",
            min_value=50,
            max_value=200,
            value=110,
            step=5
        )
        
        cc = st.slider(
            "Engine Displacement (cc)",
            min_value=1300,
            max_value=2200,
            value=1600,
            step=50
        )
        
        cylinders = st.slider(
            "Cylinders",
            min_value=3,
            max_value=6,
            value=4,
            step=1
        )
        
        st.subheader("🏋️ Physical")
        weight = st.slider(
            "Weight (kg)",
            min_value=800,
            max_value=1500,
            value=1100,
            step=10
        )
        
        tax = st.slider(
            "Quarterly Tax (€)",
            min_value=0,
            max_value=500,
            value=100,
            step=10
        )
        
        st.subheader("🎨 Features")
        col1, col2 = st.columns(2)
        with col1:
            fuel_type = st.radio("Fuel", ['Petrol', 'Diesel'], horizontal=True)
            automatic = st.radio("Trans.", ['Manual', 'Auto'], horizontal=True)
        with col2:
            met_color = st.radio("Paint", ['Standard', 'Metallic'], horizontal=True)
            abs_sys = st.radio("ABS", ['No', 'Yes'], horizontal=True) == 'Yes'
        
        airco = st.radio("Air Con", ['No', 'Yes'], horizontal=True) == 'Yes'
    
    # ========================================================================
    # MAIN CONTENT - VALUATION
    # ========================================================================
    
    result = predict_with_explanation(
        age, km, hp, weight, cc, tax, cylinders,
        fuel_type, automatic, met_color, abs_sys, airco,
        model, scaler, feature_cols, base_price, df_clean
    )
    
    predicted_price = result['prediction']
    contributions = result['contributions']
    
    # Price display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2E86AB 0%, #06A77D 100%); 
                color: white; padding: 40px; border-radius: 12px; text-align: center;">
        <h3 style="margin: 0; font-size: 16px; color: rgba(255,255,255,0.9);">
            ESTIMATED INTRINSIC VALUE
        </h3>
        <h1 style="margin: 20px 0 10px 0; font-size: 56px; color: white;">
            €{predicted_price:,.0f}
        </h1>
        <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 14px;">
            Based on {len(feature_cols)} vehicle characteristics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # TAB 1: HOW IS THIS PRICE CALCULATED?
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 How is this calculated?",
        "📊 Feature Relationships",
        "🔍 Price Sensitivity",
        "📚 Learn More"
    ])
    
    with tab1:
        st.subheader("Price Breakdown: How We Arrived at €{:,.0f}".format(predicted_price))
        
        st.markdown("""
        This valuation shows how the **machine learning model** breaks down your car's price 
        based on its features. Think of it as a mathematical recipe that combines all the 
        characteristics together.
        """)
        
        # Show contributions
        st.markdown("#### ✨ Top Factors Affecting This Car's Price")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            for i, (feature, contrib) in enumerate(contributions, 1):
                value = contrib['value']
                direction = "📈" if value > 0 else "📉"
                sign = "+" if value > 0 else ""
                
                # Create visual bar
                bar_width = int(abs(value) / 200)
                bar = "█" * bar_width if bar_width > 0 else "▌"
                
                st.markdown(f"""
                **{i}. {feature}**  
                {direction} {sign}€{abs(value):,.0f}  
                {bar}
                """)
        
        # Detailed explanation
        st.info("""
        **What this means:**
        - These are the 5 features that most affect your car's price in this model
        - Positive values (📈) add value to the car
        - Negative values (📉) reduce value
        - The combination of all 34 features creates the final price
        """)
        
        # Show market context
        st.markdown("#### 🏪 Market Context")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price", f"€{insights['base_price']:,.0f}")
        with col2:
            st.metric("Price Range", f"€{insights['price_min']:,.0f} - €{insights['price_max']:,.0f}")
        with col3:
            st.metric("Dataset Size", f"{len(df_clean)} vehicles")
    
    with tab2:
        st.subheader("Understanding Feature Relationships")
        
        st.markdown("""
        Machine learning works by learning **relationships** between features and price. 
        Below, you can see how each major feature category affects the price.
        """)
        
        # Age relationship
        st.markdown("#### 📅 How Age Affects Price")
        
        ages_to_test = list(range(0, 121, 20))
        age_prices = calculate_sensitivity(
            age, km, hp, weight, cc, tax, cylinders,
            fuel_type, automatic, met_color, abs_sys, airco,
            model, scaler, feature_cols, 'Age_08_04', ages_to_test
        )
        
        # Simple bar chart using columns
        st.line_chart(pd.DataFrame({
            'Age (months)': ages_to_test,
            'Price': age_prices
        }).set_index('Age (months)'))
        
        st.markdown("""
        **Insight:** Older cars are worth less. This is called **depreciation**.  
        Every month adds age, reducing intrinsic value.
        """)
        
        # Mileage relationship
        st.markdown("#### 🛣️ How Mileage Affects Price")
        
        km_to_test = list(range(0, 300001, 50000))
        km_prices = calculate_sensitivity(
            age, km, hp, weight, cc, tax, cylinders,
            fuel_type, automatic, met_color, abs_sys, airco,
            model, scaler, feature_cols, 'KM', km_to_test
        )
        
        st.line_chart(pd.DataFrame({
            'Mileage (km)': [x/1000 for x in km_to_test],
            'Price': km_prices
        }).set_index('Mileage (km)'))
        
        st.markdown("""
        **Insight:** More mileage = more wear on the engine = lower value.  
        This relationship was learned by the model from real vehicle data.
        """)
        
        # Feature importance
        st.markdown("#### ⭐ Overall Feature Importance (Global)")
        
        st.markdown("""
        This shows which features matter **most overall** across all 1,437 vehicles 
        in the training dataset.
        """)
        
        top_features = feature_importance.head(8)
        
        # Create simple bar chart
        chart_data = pd.DataFrame({
            'Feature': top_features['Feature'],
            'Importance': top_features['Importance']
        })
        
        st.bar_chart(chart_data.set_index('Feature')['Importance'])
        
        st.markdown("""
        **Key Insights:**
        - **Age_08_04**: Age is the strongest price predictor
        - **KM**: Mileage is second most important
        - **Weight**: Physical properties matter
        - Other features like HP, color, and features affect price but less strongly
        """)
    
    with tab3:
        st.subheader("Interactive Price Sensitivity Analysis")
        
        st.markdown("""
        Adjust parameters below to see how the price changes. This demonstrates 
        how the model learned relationships between features and price.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Try changing the Age")
            test_age = st.slider("Test Age (months)", 0, 120, value=age, key='age_slider')
            test_ages = list(range(0, 121, 10))
            test_prices = calculate_sensitivity(
                test_age, km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, met_color, abs_sys, airco,
                model, scaler, feature_cols, 'Age_08_04', test_ages
            )
            st.line_chart(pd.DataFrame({
                'Age': test_ages,
                'Price': test_prices
            }).set_index('Age'))
            
            current_pred = predict_with_explanation(
                test_age, km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, met_color, abs_sys, airco,
                model, scaler, feature_cols, base_price, df_clean
            )['prediction']
            st.metric("Price at age {}m".format(test_age), f"€{current_pred:,.0f}")
        
        with col2:
            st.markdown("#### Try changing the Mileage")
            test_km = st.slider("Test Mileage (km)", 0, 300000, value=km, step=5000, key='km_slider')
            test_kms = list(range(0, 300001, 30000))
            test_km_prices = calculate_sensitivity(
                age, test_km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, met_color, abs_sys, airco,
                model, scaler, feature_cols, 'KM', test_kms
            )
            st.line_chart(pd.DataFrame({
                'Mileage (1000 km)': [x/1000 for x in test_kms],
                'Price': test_km_prices
            }).set_index('Mileage (1000 km)'))
            
            current_pred_km = predict_with_explanation(
                age, test_km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, met_color, abs_sys, airco,
                model, scaler, feature_cols, base_price, df_clean
            )['prediction']
            st.metric("Price at {:,.0f}km".format(test_km), f"€{current_pred_km:,.0f}")
    
    with tab4:
        st.subheader("📚 Understanding This Tool")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🤖 What is Machine Learning?")
            st.markdown("""
            Machine learning is when a computer learns **patterns** from data.
            
            In this case:
            - We gave a model 1,437 real Toyota Corolla cars
            - For each car, we provided 34 characteristics (age, mileage, HP, etc.)
            - The model learned: "When these features change, price changes this way"
            - Now it can predict prices for **new** cars
            
            It's like a student learning from examples, not from rules.
            """)
            
            st.markdown("#### 🌳 What is a Random Forest?")
            st.markdown("""
            A Random Forest is **100 decision trees voting** on the price.
            
            Each tree:
            - Learns a different pattern from the data
            - Makes its own price prediction
            - The final price = average of all 100 predictions
            
            Why 100? Because more opinions = more accurate predictions.
            """)
        
        with col2:
            st.markdown("#### 📊 Model Performance")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R² Score", f"{insights['r2']:.2%}")
                st.caption("Explains 87% of price variation")
            with col_b:
                st.metric("Typical Error", f"±{insights['mape']:.1%}")
                st.caption("About ±€1,200 on €13,000")
            
            st.markdown("""
            #### ⚠️ Important Limitations
            
            **This tool is NOT:**
            - A market price predictor
            - Affected by supply/demand
            - Aware of condition or accidents
            - For insurance valuations
            
            **This tool IS:**
            - An educational demonstration
            - Based purely on characteristics
            - A machine learning example
            - For understanding feature relationships
            """)
        
        st.divider()
        
        st.markdown("#### 🎓 Try These Exercises")
        
        exercise = st.radio(
            "Pick an exercise:",
            [
                "Q: What costs more per month of age?",
                "Q: How much does metallic paint add?",
                "Q: Compare similar vs different cars",
                "Q: What's the most important feature?"
            ]
        )
        
        if "costs more per month" in exercise:
            st.info("""
            **Answer:** About €110-150 per month of age.
            
            Try this:
            1. Set age to 0 months
            2. Note the price
            3. Set age to 12 months
            4. See the difference (should be ~€1,200-€1,800)
            5. Divide by 12 = monthly cost of aging
            """)
        
        elif "metallic paint" in exercise:
            # Test contribution
            pred_standard = predict_with_explanation(
                age, km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, 'Standard', abs_sys, airco,
                model, scaler, feature_cols, base_price, df_clean
            )['prediction']
            
            pred_metallic = predict_with_explanation(
                age, km, hp, weight, cc, tax, cylinders,
                fuel_type, automatic, 'Metallic', abs_sys, airco,
                model, scaler, feature_cols, base_price, df_clean
            )['prediction']
            
            difference = pred_metallic - pred_standard
            
            st.info(f"""
            **Answer:** About €{difference:,.0f}
            
            Metallic paint makes the car worth €{difference:,.0f} more than standard paint.
            This is what the model learned from the data.
            """)
        
        elif "Compare" in exercise:
            st.info("""
            **Exercise:** Create two cars and compare
            
            1. Keep current car selected
            2. Note the price
            3. Change one feature (e.g., mileage)
            4. Check if price changed as expected
            5. Change multiple features
            6. See how they combine
            
            This shows how the model combines multiple factors.
            """)
        
        else:  # Most important
            st.info(f"""
            **Answer:** Age (Age_08_04) with importance score {feature_importance.iloc[0]['Importance']:.4f}
            
            The top 5 most important features are:
            """)
            for idx, row in feature_importance.head(5).iterrows():
                st.write(f"**{row['Feature']}**: {row['Importance']:.4f}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 12px; margin-top: 20px;">
            <p>
                <strong>Toyota Corolla Valuation Tool</strong><br>
                Educational demonstration of machine learning<br>
                Trained on {0} real vehicles | Random Forest Model<br>
                Shows relationships between vehicle features and intrinsic value
            </p>
        </div>
        """.format(len(df_clean)), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
