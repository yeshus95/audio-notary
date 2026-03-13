def compute_trust_score(layer1, layer2):
    """
    Weighted Trust Score:
      - Biological features: 60%
      - Metadata integrity:  25%
      - Splice integrity:    15%
    """

    # --- Metadata sub-score ---
    meta_score = 100
    if layer1.get('editing_software_detected'):
        meta_score -= 35
    if layer1.get('size_duration_anomaly'):
        meta_score -= 30
    meta_score = max(meta_score, 0)

    # --- Splice sub-score ---
    silence_count = layer1.get('silence_anomalies', 0)
    if silence_count == 0:
        splice_score = 100
    elif silence_count == 1:
        splice_score = 55
    elif silence_count <= 3:
        splice_score = 25
    else:
        splice_score = 5

    # --- Biological sub-score ---
    bio_score = layer2.get('biological_score', 50)

    # --- Hard override using your actual data patterns ---
    shimmer = layer2.get('shimmer') or 0
    voiced_rate = layer2.get('voiced_rate_per_sec') or 0
    spec_flatness = layer2.get('spectral_flatness') or 0
    hnr = layer2.get('hnr') or 0

    # If shimmer is low AND voiced rate is very high — almost certainly synthetic
    if shimmer < 0.55 and voiced_rate > 150:
        bio_score = max(0, bio_score - 25)
        layer2['flags'].append(
            "Combined low shimmer + high voiced rate — strong synthetic pattern detected")

    # If spectral flatness is high AND shimmer is low — synthetic
    if spec_flatness > 0.020 and shimmer < 0.60:
        bio_score = max(0, bio_score - 15)
        layer2['flags'].append(
            "High spectral flatness with low shimmer — synthetic audio pattern")

    # Cap score hard if multiple synthetic signals fire
    bio_flags = layer2.get('flags', [])
    synthetic_keywords = ['synthetic', 'TTS', 'unnaturally', 'very high voiced',
                          'high spectral', 'strong synthetic', 'erratic']
    synthetic_flag_count = sum(
        1 for f in bio_flags
        if any(kw.lower() in f.lower() for kw in synthetic_keywords)
    )
    if synthetic_flag_count >= 3:
        bio_score = max(0, min(bio_score, 25))
    elif synthetic_flag_count >= 2:
        bio_score = max(0, min(bio_score, 40))

    # --- Weighted final score ---
    trust_score = (
        bio_score    * 0.60 +
        meta_score   * 0.25 +
        splice_score * 0.15
    )
    trust_score = round(trust_score, 1)

    # --- Verdict ---
    is_edited = (
        silence_count > 0
        or layer1.get('editing_software_detected')
        or layer1.get('size_duration_anomaly')
    )
    is_biological = bio_score >= 60

    if is_biological and is_edited:
        verdict = "AUTHENTIC VOICE — POTENTIALLY EDITED"
        verdict_class = "warning"
    elif is_biological and not is_edited:
        verdict = "LIKELY AUTHENTIC"
        verdict_class = "success"
    elif not is_biological and is_edited:
        verdict = "SYNTHETIC / MANIPULATED"
        verdict_class = "danger"
    else:
        verdict = "SYNTHETIC VOICE DETECTED"
        verdict_class = "danger"

    return {
        'trust_score': trust_score,
        'verdict': verdict,
        'verdict_class': verdict_class,
        'sub_scores': {
            'metadata_integrity': meta_score,
            'biological_features': bio_score,
            'splice_integrity': splice_score
        }
    }