"""
Vector Store Test — Ingest sample ads + run searches

This script:
  1. Ingests 6 sample ad JSONs (simulating pipeline output)
  2. Runs semantic, filter, and hybrid searches
  3. Verifies retrieval correctness

Run: python test_vector_store.py
"""

import json
import time
from pipeline.vector_store import AdVectorStore


# ------------------------------------------------------------------
# Sample ad JSONs (simulating pipeline output for different brands)
# ------------------------------------------------------------------

SAMPLE_ADS = [
    # --- Nike Air Max 90 (action, running) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Nike", "parent_company": "", "sub_brand": "Air Max", "logo_detected": True, "logo_description": "Swoosh", "logo_position": "Top right", "brand_colors_hex": ["#000000", "#ffffff"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "Air Max 90", "product_category": "Footwear", "product_subcategory": "Running Shoes", "product_description": "Iconic cushioned running shoe with visible Air unit", "product_features": ["Visible Air unit", "Waffle outsole", "Padded collar"], "price_mentioned": "$130", "offer_or_discount": "", "model_or_variant": "Air Max 90"},
            "text_content": {"headline": "Just Do It", "tagline": "Just Do It", "subheadline": "", "body_copy": "", "call_to_action": "Shop Now", "fine_print": "", "contact_info": "", "hashtags": ["#JustDoIt"], "all_raw_text": ["Nike", "Just Do It", "Air Max 90"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": True, "transcript": "Push your limits. Every step counts. The Air Max 90 — built for those who never stop.", "speaker_count": 1, "has_music": True, "music_mood": "Energetic", "music_genre": "Electronic", "sound_effects": [], "voiceover_tone": "Motivational", "jingle_detected": False},
            "visual_analysis": {"scene_description": "Athletes running through urban streets at dawn, close-ups of shoes hitting pavement", "setting": "Urban cityscape", "dominant_colors_hex": ["#000000", "#FF6B00"], "color_mood": "Bold, energetic", "layout_style": "Dynamic", "people_detected": True, "people_count": 3, "people_description": "Young athletes in running gear", "celebrity_or_figure": "", "objects_detected": ["Running shoes", "City buildings", "Street"], "image_quality": "High", "aspect_ratio": "16:9"},
            "video_analysis": {"duration_seconds": 30.0, "scene_count": 4, "scenes": [], "has_subtitles": False, "transition_styles": ["Quick cut"], "pacing": "Fast-paced, energetic"},
            "classification": {"industry": "Sportswear", "ad_objective": "Product launch", "ad_style": "Lifestyle action", "target_audience": "Young athletes and runners", "target_gender": "", "target_age_range": "18-35", "emotional_appeal": "Motivation, achievement", "tone": "Bold, motivational", "themes": ["Athletics", "Achievement", "Urban lifestyle"], "seasonal_relevance": ""},
            "compliance_and_legal": {"has_disclaimer": False, "disclaimer_text": "", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": []},
            "ad_description": {"short_summary": "Nike Air Max 90 ad featuring athletes running through city streets at dawn.", "detailed_description": "A fast-paced Nike ad showcasing the Air Max 90 with athletes sprinting through urban landscapes. The ad emphasizes the shoe's performance with close-up shots of the visible Air unit hitting pavement, set to an energetic electronic soundtrack.", "creative_strategy": "Action-lifestyle showcasing product performance in real-world setting"},
        },
        "video_link": "https://example.com/nike_airmax90.mp4",
    },
    # --- Nike Find Your Greatness (emotional, inclusive) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Nike", "parent_company": "", "sub_brand": "", "logo_detected": True, "logo_description": "Swoosh", "logo_position": "Center", "brand_colors_hex": ["#000000"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "Nike Training", "product_category": "Apparel", "product_subcategory": "Training Apparel", "product_description": "Training gear for everyday athletes", "product_features": ["Dri-FIT technology", "Breathable fabric"], "price_mentioned": "", "offer_or_discount": "", "model_or_variant": ""},
            "text_content": {"headline": "Find Your Greatness", "tagline": "Find Your Greatness", "subheadline": "", "body_copy": "", "call_to_action": "", "fine_print": "", "contact_info": "", "hashtags": [], "all_raw_text": ["Nike", "Find Your Greatness"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": True, "transcript": "Greatness is not some rare DNA strand. It's not in one special place. Greatness is wherever somebody is trying to find it.", "speaker_count": 1, "has_music": True, "music_mood": "Inspirational", "music_genre": "Orchestral", "sound_effects": [], "voiceover_tone": "Inspirational", "jingle_detected": False},
            "visual_analysis": {"scene_description": "Everyday people exercising in various settings — a boy jogging on a quiet road, a woman doing yoga in a park", "setting": "Various everyday locations", "dominant_colors_hex": ["#333333"], "color_mood": "Warm, hopeful", "layout_style": "Documentary", "people_detected": True, "people_count": 5, "people_description": "Ordinary people of various ages exercising", "celebrity_or_figure": "", "objects_detected": ["Running shoes", "Yoga mat", "Park"], "image_quality": "High", "aspect_ratio": "16:9"},
            "video_analysis": {"duration_seconds": 60.0, "scene_count": 8, "scenes": [], "has_subtitles": False, "transition_styles": ["Fade", "Cut"], "pacing": "Slow, emotional"},
            "classification": {"industry": "Sportswear", "ad_objective": "Brand awareness", "ad_style": "Emotional storytelling", "target_audience": "General fitness enthusiasts", "target_gender": "", "target_age_range": "18-45", "emotional_appeal": "Inspiration, inclusivity", "tone": "Inspirational, warm", "themes": ["Inclusivity", "Motivation", "Everyday heroism"], "seasonal_relevance": ""},
            "compliance_and_legal": {"has_disclaimer": False, "disclaimer_text": "", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": []},
            "ad_description": {"short_summary": "Nike inspirational ad showing everyday people finding greatness through exercise.", "detailed_description": "A documentary-style Nike ad featuring ordinary people of various ages and backgrounds exercising in everyday settings. The powerful voiceover emphasizes that greatness isn't reserved for elite athletes but is found wherever someone tries.", "creative_strategy": "Emotional storytelling emphasizing inclusivity and everyday achievement"},
        },
        "video_link": "https://example.com/nike_find_greatness.mp4",
    },
    # --- Benjamin Moore Regal Select (family, heartwarming) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Benjamin Moore", "parent_company": "", "sub_brand": "", "logo_detected": True, "logo_description": "Triangle logo", "logo_position": "Multiple frames", "brand_colors_hex": ["#ffffff"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "Regal Select Paint", "product_category": "Paint", "product_subcategory": "Interior Paint", "product_description": "Long-Lasting Finish to Easily Clean Stains and Scuffs", "product_features": ["Matte Finish", "Acrylic Paint & Primer"], "price_mentioned": "", "offer_or_discount": "", "model_or_variant": "Regal Select"},
            "text_content": {"headline": "", "tagline": "See the Love", "subheadline": "", "body_copy": "", "call_to_action": "", "fine_print": "", "contact_info": "", "hashtags": [], "all_raw_text": ["Benjamin Moore", "Regal Select", "See the Love"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": True, "transcript": "Third 10, 9, 8, 10, so it's 8, 5, 4, 3, 2, 1. Here I come. Yay! You're actually going. Yay! Yay! Look at me. Can I show you something? Not now. You're going to come visit me, right? I mean if you're lucky. Happy, happy, ready, ready.", "speaker_count": 1, "has_music": False, "music_mood": "", "music_genre": "", "sound_effects": [], "voiceover_tone": "Playful", "jingle_detected": False},
            "visual_analysis": {"scene_description": "The video shows various scenes of painting and family moments.", "setting": "Home and outdoors", "dominant_colors_hex": ["#87CEEB"], "color_mood": "Calming, playful", "layout_style": "Mixed", "people_detected": True, "people_count": 2, "people_description": "A man and a young girl", "celebrity_or_figure": "", "objects_detected": ["Paint cans", "Paintbrushes", "School bus"], "image_quality": "High", "aspect_ratio": "Variable"},
            "video_analysis": {"duration_seconds": 75.0, "scene_count": 5, "scenes": [], "has_subtitles": False, "transition_styles": ["Fade", "Cut"], "pacing": "Emotional, heartwarming"},
            "classification": {"industry": "Paints and Coatings", "ad_objective": "Brand awareness and emotional connection", "ad_style": "Emotional storytelling", "target_audience": "Homeowners and parents", "target_gender": "", "target_age_range": "", "emotional_appeal": "Love, happiness", "tone": "Playful, heartwarming", "themes": ["Family", "Love", "Home"], "seasonal_relevance": ""},
            "compliance_and_legal": {"has_disclaimer": False, "disclaimer_text": "", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": []},
            "ad_description": {"short_summary": "The ad showcases Benjamin Moore's Regal Select Paint and emphasizes the joy of home and family.", "detailed_description": "The video ad for Benjamin Moore's Regal Select Paint features a mix of scenes showcasing the product and heartwarming family moments. It starts with product shots of the paint cans, followed by a scene of painting a room. The narrative then shifts to a playful countdown and a heartwarming outdoor scene of a man and a young girl. The ad concludes with the Benjamin Moore logo and the tagline 'See the Love'.", "creative_strategy": "Emotional storytelling through family moments and product showcase"},
        },
        "video_link": "https://example.com/benjamin_moore_see_the_love.mp4",
    },
    # --- Apple iPhone 15 Pro (sleek, premium) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Apple", "parent_company": "", "sub_brand": "", "logo_detected": True, "logo_description": "Apple logo", "logo_position": "End frame", "brand_colors_hex": ["#000000", "#ffffff"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "iPhone 15 Pro", "product_category": "Electronics", "product_subcategory": "Smartphones", "product_description": "Titanium design with A17 Pro chip", "product_features": ["Titanium frame", "48MP camera", "Action Button"], "price_mentioned": "$999", "offer_or_discount": "", "model_or_variant": "iPhone 15 Pro"},
            "text_content": {"headline": "Titanium", "tagline": "Titanium. So Strong. So Light. So Pro.", "subheadline": "", "body_copy": "", "call_to_action": "Learn More", "fine_print": "", "contact_info": "", "hashtags": [], "all_raw_text": ["iPhone 15 Pro", "Titanium"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": False, "transcript": "", "speaker_count": 0, "has_music": True, "music_mood": "Sleek, modern", "music_genre": "Electronic ambient", "sound_effects": ["Metallic clinks", "Whoosh"], "voiceover_tone": "", "jingle_detected": False},
            "visual_analysis": {"scene_description": "Cinematic shots of the titanium iPhone rotating in space with dramatic lighting, close-ups of the camera system and titanium edges", "setting": "Studio/abstract", "dominant_colors_hex": ["#1C1C1E", "#C0C0C0"], "color_mood": "Premium, sleek", "layout_style": "Product showcase", "people_detected": False, "people_count": 0, "people_description": "", "celebrity_or_figure": "", "objects_detected": ["iPhone", "Titanium frame"], "image_quality": "Very High", "aspect_ratio": "16:9"},
            "video_analysis": {"duration_seconds": 45.0, "scene_count": 6, "scenes": [], "has_subtitles": False, "transition_styles": ["Morph", "Zoom"], "pacing": "Sleek, measured"},
            "classification": {"industry": "Consumer Electronics", "ad_objective": "Product launch", "ad_style": "Product showcase", "target_audience": "Tech enthusiasts and premium consumers", "target_gender": "", "target_age_range": "25-55", "emotional_appeal": "Premium, innovation", "tone": "Sleek, confident", "themes": ["Innovation", "Premium design", "Technology"], "seasonal_relevance": ""},
            "compliance_and_legal": {"has_disclaimer": False, "disclaimer_text": "", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": []},
            "ad_description": {"short_summary": "Apple iPhone 15 Pro product showcase highlighting the titanium design.", "detailed_description": "A cinematic Apple ad showcasing the iPhone 15 Pro's titanium construction. The ad features dramatic studio lighting with the phone rotating to reveal its titanium edges, camera system, and premium build quality. No voiceover — only sleek electronic music and metallic sound effects.", "creative_strategy": "Premium product showcase with cinematic photography and material-focused storytelling"},
        },
        "video_link": "https://example.com/apple_iphone15pro.mp4",
    },
    # --- Coca-Cola Summer (joyful, beach, friends) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Coca-Cola", "parent_company": "The Coca-Cola Company", "sub_brand": "", "logo_detected": True, "logo_description": "Classic Coca-Cola script", "logo_position": "End frame", "brand_colors_hex": ["#FF0000", "#ffffff"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "Coca-Cola Classic", "product_category": "Beverages", "product_subcategory": "Soft Drinks", "product_description": "The original taste of Coca-Cola", "product_features": ["Classic formula"], "price_mentioned": "", "offer_or_discount": "", "model_or_variant": "Classic"},
            "text_content": {"headline": "", "tagline": "Taste the Feeling", "subheadline": "", "body_copy": "", "call_to_action": "", "fine_print": "", "contact_info": "", "hashtags": ["#TasteTheFeeling"], "all_raw_text": ["Coca-Cola", "Taste the Feeling"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": False, "transcript": "", "speaker_count": 0, "has_music": True, "music_mood": "Joyful, summery", "music_genre": "Pop", "sound_effects": ["Bottle opening", "Fizzing"], "voiceover_tone": "", "jingle_detected": True},
            "visual_analysis": {"scene_description": "Friends sharing Coca-Cola at a summer beach party, close-ups of ice-cold bottles with condensation", "setting": "Beach", "dominant_colors_hex": ["#FF0000", "#FFD700"], "color_mood": "Vibrant, warm", "layout_style": "Lifestyle", "people_detected": True, "people_count": 6, "people_description": "Young friends laughing and enjoying drinks at a beach", "celebrity_or_figure": "", "objects_detected": ["Coca-Cola bottles", "Beach", "Cooler", "Sunglasses"], "image_quality": "High", "aspect_ratio": "16:9"},
            "video_analysis": {"duration_seconds": 30.0, "scene_count": 5, "scenes": [], "has_subtitles": False, "transition_styles": ["Cut", "Slow motion"], "pacing": "Upbeat, summery"},
            "classification": {"industry": "Beverages", "ad_objective": "Brand reinforcement", "ad_style": "Lifestyle", "target_audience": "Young adults", "target_gender": "", "target_age_range": "16-30", "emotional_appeal": "Joy, friendship", "tone": "Joyful, vibrant", "themes": ["Friendship", "Summer", "Refreshment"], "seasonal_relevance": "Summer"},
            "compliance_and_legal": {"has_disclaimer": False, "disclaimer_text": "", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": []},
            "ad_description": {"short_summary": "Coca-Cola summer beach ad showing friends enjoying ice-cold bottles together.", "detailed_description": "A vibrant Coca-Cola ad set at a summer beach party. Young friends share ice-cold Coca-Cola bottles while laughing and enjoying the sunshine. The ad features close-up shots of condensation on the bottles and the iconic fizz, set to upbeat pop music.", "creative_strategy": "Lifestyle association connecting Coca-Cola with joyful summer moments and friendship"},
        },
        "video_link": "https://example.com/cocacola_summer.mp4",
    },
    # --- Toyota Camry (reliable, family sedan) ---
    {
        "json": {
            "_meta": {"schema_version": "1.0.0", "input_media_type": "VIDEO"},
            "source": {"medium": "video", "platform": "", "format": "", "estimated_era": ""},
            "brand": {"company_name": "Toyota", "parent_company": "", "sub_brand": "", "logo_detected": True, "logo_description": "Toyota emblem", "logo_position": "End frame", "brand_colors_hex": ["#EB0A1E", "#ffffff"], "brand_url": "", "brand_social_handles": []},
            "product": {"product_name": "Camry 2025", "product_category": "Automobiles", "product_subcategory": "Sedans", "product_description": "Reliable family sedan with hybrid efficiency", "product_features": ["Hybrid powertrain", "Toyota Safety Sense", "Spacious interior"], "price_mentioned": "$28,855", "offer_or_discount": "0% APR for 60 months", "model_or_variant": "Camry XLE Hybrid"},
            "text_content": {"headline": "Let's Go Places", "tagline": "Let's Go Places", "subheadline": "The all-new Camry Hybrid", "body_copy": "", "call_to_action": "Visit your local dealer", "fine_print": "0% APR for qualified buyers", "contact_info": "", "hashtags": [], "all_raw_text": ["Toyota", "Camry", "Let's Go Places"]},
            "language": {"primary_language": "English", "primary_language_name": "English", "secondary_languages": [], "is_multilingual": False},
            "audio": {"has_speech": True, "transcript": "Life doesn't wait. Neither should you. The all-new Camry Hybrid — efficiency meets style. Let's go places.", "speaker_count": 1, "has_music": True, "music_mood": "Uplifting", "music_genre": "Indie pop", "sound_effects": ["Engine hum"], "voiceover_tone": "Warm, confident", "jingle_detected": False},
            "visual_analysis": {"scene_description": "A family loading up the Camry for a road trip, driving through scenic highways, arriving at a lakeside destination", "setting": "Suburban home, highway, lakeside", "dominant_colors_hex": ["#C0C0C0", "#4682B4"], "color_mood": "Warm, adventurous", "layout_style": "Narrative", "people_detected": True, "people_count": 4, "people_description": "A family of four — parents and two children", "celebrity_or_figure": "", "objects_detected": ["Toyota Camry", "Suitcases", "Highway", "Lake"], "image_quality": "High", "aspect_ratio": "16:9"},
            "video_analysis": {"duration_seconds": 45.0, "scene_count": 5, "scenes": [], "has_subtitles": False, "transition_styles": ["Cut", "Dissolve"], "pacing": "Moderate, warm"},
            "classification": {"industry": "Automotive", "ad_objective": "Product launch", "ad_style": "Narrative lifestyle", "target_audience": "Families and practical car buyers", "target_gender": "", "target_age_range": "30-55", "emotional_appeal": "Reliability, adventure", "tone": "Warm, confident", "themes": ["Family", "Road trips", "Reliability"], "seasonal_relevance": ""},
            "compliance_and_legal": {"has_disclaimer": True, "disclaimer_text": "0% APR for qualified buyers. See dealer for details.", "has_age_restriction": False, "age_restriction_note": "", "has_trademark_symbols": True, "copyright_notice": "", "regulatory_body_mentioned": ""},
            "engagement_elements": {"has_qr_code": False, "qr_code_content": "", "has_coupon_code": False, "coupon_code": "", "has_social_proof": False, "social_proof_type": "", "urgency_elements": ["Limited time offer"]},
            "ad_description": {"short_summary": "Toyota Camry Hybrid ad showing a family road trip from suburbs to a scenic lake.", "detailed_description": "A warm narrative ad for the Toyota Camry Hybrid. A family of four packs up and drives through scenic highways to a lakeside destination. The ad highlights the car's hybrid efficiency, spacious interior, and reliability, with a voiceover emphasizing that life doesn't wait.", "creative_strategy": "Narrative lifestyle ad connecting the Camry with family adventure and dependability"},
        },
        "video_link": "https://example.com/toyota_camry_hybrid.mp4",
    },
]


def print_results(results: list[dict], show_link: bool = True):
    """Pretty-print search results."""
    if not results:
        print("    (no results)")
        return
    for r in results:
        p = r["payload"]
        brand = p.get("company_name", "?")
        product = p.get("product_name", "?")
        tone = p.get("tone", "")
        style = p.get("ad_style", "")
        print(f"    [{r['score']:.3f}] {brand} — {product}  |  {style}, {tone}")
        if show_link:
            print(f"            → {p.get('video_link', 'N/A')}")


def main():
    print("=" * 60)
    print("AD VECTOR STORE — INGEST + SEARCH TEST")
    print("=" * 60)

    # Initialize (creates data/qdrant_db/ on first run)
    store = AdVectorStore()

    # Clear for clean test
    store.clear_all()
    print(f"\nStore cleared. Count: {store.count()}")

    # ------------------------------------------------------------------
    # 1. INGEST
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("PHASE 1: INGESTING ADS")
    print(f"{'─' * 50}")

    t0 = time.perf_counter()
    for i, ad in enumerate(SAMPLE_ADS):
        ad_id = store.add_ad(ad["json"], video_link=ad["video_link"])
        brand = ad["json"]["brand"]["company_name"]
        product = ad["json"]["product"]["product_name"]
        print(f"  [{i+1}] {brand} — {product} → {ad_id[:12]}...")
    ingest_time = time.perf_counter() - t0

    print(f"\n  Total ads: {store.count()}")
    print(f"  Brands: {store.get_all_brands()}")
    print(f"  Ingest time: {ingest_time:.2f}s")

    # ------------------------------------------------------------------
    # 2. SEMANTIC SEARCH (no filters, just natural language)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("PHASE 2: SEMANTIC SEARCH")
    print(f"{'─' * 50}")

    semantic_tests = [
        ("emotional family ad",          "Benjamin Moore"),
        ("running shoes urban city",      "Nike"),
        ("premium technology showcase",   "Apple"),
        ("summer friends refreshing",     "Coca-Cola"),
        ("reliable family car road trip", "Toyota"),
        ("inspirational everyday people", "Nike"),
    ]

    for query, expected_top in semantic_tests:
        print(f"\n  Query: \"{query}\"")
        t0 = time.perf_counter()
        results = store.search(query=query, n_results=3)
        elapsed = time.perf_counter() - t0

        print_results(results, show_link=False)

        top = results[0]["payload"].get("company_name", "") if results else ""
        status = "✓" if top == expected_top else "✗"
        print(f"    {status} Expected: {expected_top} | Got: {top} ({elapsed*1000:.0f}ms)")

    # ------------------------------------------------------------------
    # 3. PURE FILTER SEARCH (exact metadata, no semantic)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("PHASE 3: PURE FILTER SEARCH")
    print(f"{'─' * 50}")

    # All Nike ads
    print(f"\n  Filter: company_name = Nike")
    results = store.search(query="", filters={"company_name": "Nike"})
    print_results(results, show_link=False)
    print(f"    → Found {len(results)} Nike ads (expected 2)")

    # All Sportswear industry ads
    print(f"\n  Filter: industry = Sportswear")
    results = store.search(query="", filters={"industry": "Sportswear"})
    print_results(results, show_link=False)
    print(f"    → Found {len(results)} Sportswear ads (expected 2)")

    # All Beverages
    print(f"\n  Filter: industry = Beverages")
    results = store.search(query="", filters={"industry": "Beverages"})
    print_results(results, show_link=False)
    print(f"    → Found {len(results)} Beverages ads (expected 1)")

    # ------------------------------------------------------------------
    # 4. HYBRID SEARCH (filters + semantic)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("PHASE 4: HYBRID SEARCH (filter + semantic)")
    print(f"{'─' * 50}")

    # Nike + emotional → should rank "Find Your Greatness" above Air Max
    print(f"\n  Hybrid: \"inspirational emotional\" + company_name=Nike")
    results = store.search_hybrid("inspirational emotional", brand="Nike")
    print_results(results, show_link=False)

    # Sportswear + shoes → should get Air Max
    print(f"\n  Hybrid: \"running shoes\" + industry=Sportswear")
    results = store.search_hybrid("running shoes performance", industry="Sportswear")
    print_results(results, show_link=False)

    # Emotional storytelling style across all brands
    print(f"\n  Hybrid: \"family love\" + ad_style=Emotional storytelling")
    results = store.search_hybrid("family love", ad_style="Emotional storytelling")
    print_results(results, show_link=False)

    # ------------------------------------------------------------------
    # 5. METADATA VALUES (for Streamlit dropdowns)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("PHASE 5: AVAILABLE FILTER VALUES")
    print(f"{'─' * 50}")

    meta = store.get_all_metadata_values()
    for field, values in meta.items():
        print(f"  {field}: {sorted(values)}")

    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ALL TESTS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()