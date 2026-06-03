from flask import Flask, request, render_template_string
from transformers import pipeline
import datetime
import plotly.graph_objs as go
import plotly.utils
import json
import pandas as pd

app = Flask(__name__)

# In-memory storage for reviews (no MongoDB needed)
reviews = []

# Load the sentiment analysis model from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# HTML template with form and chart
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Sentiment Analyzer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        textarea { width: 100%; padding: 10px; font-size: 16px; }
        input[type="submit"] { margin-top: 10px; padding: 8px 16px; font-size: 16px; }
        .result { margin-top: 20px; padding: 15px; border-radius: 5px; font-size: 18px; }
        .positive { background-color: #d4edda; color: #155724; border-left: 5px solid #28a745; }
        .negative { background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
        .neutral { background-color: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }
        h1, h2 { color: #333; }
    </style>
</head>
<body>
    <h1>🧠 AI Sentiment Analyzer</h1>
    <p>Enter a product review or social media comment below:</p>
    <form method="POST">
        <textarea name="text" rows="4" placeholder="Type your text here..."></textarea><br>
        <input type="submit" value="Analyze Sentiment">
    </form>

    {% if sentiment %}
    <div class="result {{ class_name }}">
        <strong>Sentiment:</strong> {{ sentiment }}<br>
        <strong>Confidence:</strong> {{ confidence }}%
    </div>
    {% endif %}

    <h2>📊 Sentiment Distribution Report</h2>
    <div id="chart"></div>
    <script>
        var chartData = {{ chart_json|safe }};
        Plotly.newPlot('chart', chartData.data, chartData.layout);
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    confidence = None
    class_name = ""

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()
        if user_text:
            # Run sentiment analysis
            result = sentiment_pipeline(user_text)[0]
            label = result['label']   # POSITIVE or NEGATIVE
            score = result['score'] * 100

            # Determine sentiment with neutral threshold
            if label == "POSITIVE":
                if score >= 60:
                    sentiment = "Positive 😊"
                    class_name = "positive"
                else:
                    sentiment = "Neutral 😐"
                    class_name = "neutral"
            else:  # NEGATIVE
                if score >= 60:
                    sentiment = "Negative 😞"
                    class_name = "negative"
                else:
                    sentiment = "Neutral 😐"
                    class_name = "neutral"

            confidence = round(score, 1)

            # Store review
            reviews.append({
                "sentiment": sentiment,
                "confidence": score,
                "text": user_text,
                "timestamp": datetime.datetime.now()
            })

    # Generate chart from stored reviews
    if reviews:
        df = pd.DataFrame(reviews)
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        fig = go.Figure(data=[
            go.Bar(
                x=sentiment_counts['sentiment'],
                y=sentiment_counts['count'],
                marker_color=['#28a745' if s == 'Positive 😊' else '#dc3545' if s == 'Negative 😞' else '#ffc107' for s in sentiment_counts['sentiment']]
            )
        ])
        fig.update_layout(
            title="Distribution of Analyzed Sentiments",
            xaxis_title="Sentiment",
            yaxis_title="Number of Reviews",
            template="plotly_white"
        )
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        # Empty chart placeholder
        fig = go.Figure()
        fig.add_annotation(text="No reviews analyzed yet. Submit some text above.", showarrow=False)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, confidence=confidence, class_name=class_name, chart_json=chart_json)

if __name__ == "__main__":
    app.run(debug=True)