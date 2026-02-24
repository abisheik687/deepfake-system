/**
 * KAVACH.AI Extension — Background Service Worker v1.2
 *
 * SIMPLIFIED: Background is now ONLY used for AI explanation API calls
 * (Gemini / OpenAI). Image analysis is handled DIRECTLY in content.js
 * via fetch — no more message-passing bottleneck.
 */

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'EXPLAIN_FAKE') {
        handleExplainFake(msg)
            .then(sendResponse)
            .catch(err => sendResponse({ status: 'error', explanation: err.message }));
        return true;
    }
});

async function handleExplainFake({ imageUrl, sourcePage, caption, verdict, confidence }) {
    const { apiKey, apiProvider } = await chrome.storage.local.get(['apiKey', 'apiProvider']);

    if (!apiKey) {
        return {
            status: 'no_key',
            explanation: 'Add your Gemini or OpenAI API key in the extension popup to get explanations.'
        };
    }

    const provider = apiProvider || 'gemini';
    const confidencePct = Math.round(confidence * 100);
    const prompt = `A deepfake AI flagged an image from ${sourcePage} as ${verdict} with ${confidencePct}% confidence. ${caption ? `Caption: "${caption.slice(0, 120)}"` : ''} Explain in 2-3 sentences why this content may be fake or manipulated. Search the web for any related news about deepfakes on this platform.`;

    try {
        if (provider === 'gemini') {
            const res = await fetch(
                `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        contents: [{ parts: [{ text: prompt }] }],
                        tools: [{ google_search: {} }],
                        generationConfig: { temperature: 0.3, maxOutputTokens: 300 }
                    })
                }
            );
            const data = await res.json();
            const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || 'No explanation available.';
            return { status: 'ok', provider: 'gemini', explanation: text.trim() };

        } else {
            const res = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
                body: JSON.stringify({
                    model: 'gpt-4o-mini',
                    messages: [
                        { role: 'system', content: 'You are a deepfake analyst. Be concise and factual.' },
                        { role: 'user', content: prompt }
                    ],
                    max_tokens: 200
                })
            });
            const data = await res.json();
            const text = data?.choices?.[0]?.message?.content || 'No explanation available.';
            return { status: 'ok', provider: 'openai', explanation: text.trim() };
        }
    } catch (err) {
        return { status: 'error', explanation: `AI error: ${err.message}` };
    }
}
