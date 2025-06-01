// static/js/modules/api.js

export async function fetchApi(endpoint, method = 'GET', body = null) {
    const options = { method };
    if (body) {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(body);
    }
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            const errorData = await response.json();
            console.error(`API Error (${response.status}) for ${endpoint}:`, errorData.error || response.statusText);
            alert(`Error: ${errorData.error || response.statusText}`);
            return null;
        }
        return await response.json();
    } catch (error) {
        console.error('Network or API call failed:', error);
        alert(`Network error: ${error.message}`);
        return null;
    }
}