from flask import Flask, request, jsonify, render_template, send_file
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- WEKA CENTROIDS ----------------
WEKA_CENTROIDS = np.array([
    [0.8019, 0.7607],
    [0.4490, 0.2346],
    [0.4379, 0.7475],
    [0.8189, 0.2666]
])

# ---------------- SMART MODEL SELECTION ----------------
def should_retrain(X_scaled, threshold=0.15):
    original_mean = np.array([0.6178, 0.5068])
    original_std  = np.array([0.25, 0.25])

    new_mean = np.mean(X_scaled, axis=0)
    new_std  = np.std(X_scaled, axis=0)

    mean_diff = np.linalg.norm(new_mean - original_mean)
    std_diff  = np.linalg.norm(new_std - original_std)

    print("Mean diff:", mean_diff)
    print("Std diff:", std_diff)

    return mean_diff > threshold or std_diff > threshold


def predict_with_weka_centroids(X_scaled):
    kmeans = KMeans(n_clusters=4, init=WEKA_CENTROIDS, n_init=1, max_iter=1, random_state=42)
    kmeans.fit(X_scaled)
    kmeans.cluster_centers_ = WEKA_CENTROIDS
    return kmeans.predict(X_scaled)


def train_new_model(X_scaled):
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans.cluster_centers_


# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # Load data
        try:
            customer_data = pd.read_csv(filepath)
        except UnicodeDecodeError:
            customer_data = pd.read_csv(filepath, encoding='unicode_escape')

        # Season handling
        if 'Season' in customer_data.columns:
            season_mapping = {'Summer': 0, 'Winter': 1, 'Spring': 2, 'Autumn': 3, 'Monsoon': 4}
            customer_data['Season_Number'] = customer_data['Season'].map(season_mapping)
        else:
            customer_data['Season'] = 'Unknown'
            customer_data['Season_Number'] = 0

        # Date handling
        if 'InvoiceDate' in customer_data.columns:
            customer_data['Purchase_Date'] = pd.to_datetime(customer_data['InvoiceDate'])
        else:
            customer_data['Purchase_Date'] = pd.to_datetime(customer_data['Purchase_Date'])

        today = datetime.now()
        customer_data['Recency'] = (today - customer_data['Purchase_Date']).dt.days

        # Price calculations
        if 'UnitPrice' in customer_data.columns and 'Quantity' in customer_data.columns:
            customer_data['Total_Price'] = customer_data['Quantity'] * customer_data['UnitPrice']

        customer_data['Avg_Order_Value'] = (
            customer_data['Total_Price'] / customer_data['Quantity'].replace(0, 1)
        )

        # Clean data
        customer_data = customer_data.dropna(subset=['Recency', 'Avg_Order_Value']).copy()

        if 'Customer_ID' not in customer_data.columns:
            customer_data['Customer_ID'] = customer_data.get('CustomerID', customer_data.index)

        # Scaling
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(customer_data[['Recency', 'Avg_Order_Value']])

        # ---------------- SMART MODEL DECISION ----------------
        if should_retrain(X_scaled):
            labels, new_centroids = train_new_model(X_scaled)
            customer_data['Cluster_Label'] = labels
            print("Using NEW trained model")
        else:
            customer_data['Cluster_Label'] = predict_with_weka_centroids(X_scaled)
            print("Using EXISTING Weka model")

        print("Clusters found:", np.unique(customer_data['Cluster_Label']))

        # ---------------- SEGMENTATION ----------------
        cluster_summary = customer_data.groupby('Cluster_Label').agg({
            'Recency': 'mean',
            'Avg_Order_Value': 'mean'
        }).reset_index()

        segment_mapping = {}

        cluster_summary = cluster_summary.sort_values(by='Avg_Order_Value', ascending=False)

        if len(cluster_summary) >= 4:
            high_spenders = cluster_summary.iloc[0:2].sort_values(by='Recency', ascending=True)
            low_spenders  = cluster_summary.iloc[2:4].sort_values(by='Recency', ascending=True)

            segment_mapping[high_spenders.iloc[0]['Cluster_Label']] = "High-Value Loyal Customers"
            segment_mapping[high_spenders.iloc[1]['Cluster_Label']] = "Seasonal Buyers"
            segment_mapping[low_spenders.iloc[0]['Cluster_Label']]  = "Low-Value Frequent Buyers"
            segment_mapping[low_spenders.iloc[1]['Cluster_Label']]  = "Lost Customers"
        else:
            median_recency = cluster_summary['Recency'].median()
            median_value   = cluster_summary['Avg_Order_Value'].median()

            for _, row in cluster_summary.iterrows():
                if row['Avg_Order_Value'] >= median_value and row['Recency'] <= median_recency:
                    segment_mapping[row['Cluster_Label']] = "High-Value Loyal Customers"
                elif row['Avg_Order_Value'] < median_value and row['Recency'] <= median_recency:
                    segment_mapping[row['Cluster_Label']] = "Low-Value Frequent Buyers"
                elif row['Avg_Order_Value'] >= median_value and row['Recency'] > median_recency:
                    segment_mapping[row['Cluster_Label']] = "Seasonal Buyers"
                else:
                    segment_mapping[row['Cluster_Label']] = "Lost Customers"

        customer_data['Segment'] = customer_data['Cluster_Label'].map(segment_mapping)
        customer_data['Segment'] = customer_data['Segment'].fillna("Seasonal Buyers")

        # Campaign mapping
        campaign_mapping = {
            "Low-Value Frequent Buyers": "Discount Coupons & Loyalty Programs",
            "High-Value Loyal Customers": "Exclusive Membership & VIP Rewards",
            "Lost Customers": "Standard Re-engagement Emails",
            "Seasonal Buyers": "Seasonal Promotions & Personalized Offers"
        }

        customer_data['Suggested_Campaign'] = customer_data['Segment'].map(campaign_mapping)

        # Save output
        output_path = os.path.join(UPLOAD_FOLDER, 'output.csv')
        customer_data.to_csv(output_path, index=False)

        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        session_path = os.path.join(UPLOAD_FOLDER, f'session_{session_id}.csv')
        customer_data.to_csv(session_path, index=False)

        return jsonify({
            'message': 'File processed successfully!',
            'download_url': '/download',
            'session_id': session_id,
            'visualization_url': f'/visualization/{session_id}'
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download')
def download_file():
    path = os.path.join(UPLOAD_FOLDER, 'output.csv')
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/visualization/<session_id>')
def visualization(session_id):
    return render_template("visualization.html", session_id=session_id)


@app.route('/api/segment-counts/<session_id>')
def segment_counts(session_id):
    path = os.path.join(UPLOAD_FOLDER, f'session_{session_id}.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'Session not found'}), 404

    data = pd.read_csv(path)
    counts = data['Segment'].value_counts().to_dict()

    return jsonify({'labels': list(counts.keys()), 'values': list(counts.values())})


@app.route('/api/spending-by-segment/<session_id>')
def spending_by_segment(session_id):
    path = os.path.join(UPLOAD_FOLDER, f'session_{session_id}.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'Session not found'}), 404

    data = pd.read_csv(path)
    avg_spending = data.groupby('Segment')['Total_Price'].mean().to_dict()

    return jsonify({'labels': list(avg_spending.keys()), 'values': list(avg_spending.values())})


@app.route('/api/recency-value-scatter/<session_id>')
def recency_value_scatter(session_id):
    path = os.path.join(UPLOAD_FOLDER, f'session_{session_id}.csv')
    if not os.path.exists(path):
        return jsonify({'error': 'Session not found'}), 404

    data = pd.read_csv(path)
    scatter = []

    for segment in data['Segment'].unique():
        seg = data[data['Segment'] == segment]
        scatter.append({
            'name': segment,
            'data': seg[['Recency', 'Avg_Order_Value', 'Customer_ID']].values.tolist()
        })

    return jsonify(scatter)


@app.route('/api/seasonal-distribution/<session_id>')
def seasonal_distribution(session_id):
    data = pd.read_csv(f'uploads/session_{session_id}.csv')

    # If Season column doesn't exist → create it
    if 'Season' not in data.columns:
        data['Purchase_Date'] = pd.to_datetime(data['Purchase_Date'])
        data['Month'] = data['Purchase_Date'].dt.month

        def get_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        data['Season'] = data['Month'].apply(get_season)

    result = []

    for segment in data['Segment'].unique():
        temp = data[data['Segment'] == segment]
        counts = temp['Season'].value_counts()

        result.append({
            "label": segment,
            "data": [
                    int(counts.get("Winter", 0)),
                    int(counts.get("Spring", 0)),
                    int(counts.get("Summer", 0)),
                    int(counts.get("Autumn", 0))
]
        })

    return jsonify({
        "labels": ["Winter", "Spring", "Summer", "Autumn"],
        "datasets": result
    })

if __name__ == '__main__':
    app.run(debug=True)