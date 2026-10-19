import os
from flask import Flask, render_template, request, jsonify
from predict import ScamPredictor

app = Flask(__name__, static_folder='static', template_folder='templates')
predictor = ScamPredictor()

@app.route('/')
def home():
    return render_template('index.html')

BASE_RECOMMENDATIONS = [
    'Avoid clicking links or downloading attachments from unknown senders.',
    'Do not provide personal, account, or financial information.',
    'Verify the sender through an official channel (e.g. the number on your bank card) before responding.'
]

INDICATOR_TIPS = {
    'urgency_keywords': 'This message pressures you to act "now" — legitimate organizations rarely demand instant action.',
    'fake_fee_mentions': 'It asks for a fee, deposit, or payment — a common scam tactic to extract money upfront.',
    'emotional_manipulation': 'It leans on fear, guilt, or reward to cloud your judgment — pause before reacting emotionally.',
    'isolation_attempts': 'It asks you to keep this secret or avoid telling others — a major red flag; scammers isolate victims from people who could warn them.',
    'authority_claims': 'It claims to be a bank, government agency, or law enforcement — verify independently using an official phone number or website, never one provided in the message.',
    'threat_indicators': 'It threatens a negative consequence (suspension, fines, arrest) — real institutions do not threaten you over text or email.'
}


def build_recommendations(indicators):
    tips = [INDICATOR_TIPS[key] for key in INDICATOR_TIPS if indicators.get(key)]
    return tips + BASE_RECOMMENDATIONS


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or data.get('scam_message') or '').strip()
    user_response = (data.get('user_response') or '').strip()
    metadata = data.get('metadata') or {}
    if not isinstance(metadata, dict):
        metadata = {}

    if not message:
        return jsonify({'error': 'Please provide a message to analyze.'}), 400

    result = predictor.predict(message, user_response, metadata)

    response = {
        'risk_assessment': result['risk_assessment'],
        'response_effectiveness': result['response_effectiveness'],
        'analysis': result['analysis'],
        'message': message,
        'user_response': user_response,
        'recommendations': build_recommendations(result['analysis']['metadata'])
    }
    return jsonify(response)

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host='127.0.0.1', port=port)
