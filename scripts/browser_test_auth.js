// Test frontend auth from browser console
// Open browser console (F12) and paste this code

const testLogin = async () => {
    const formData = new FormData();
    formData.append('username', 'admin@kavach.ai');
    formData.append('password', 'Kavach@2026');

    console.log('Testing login to http://localhost:8000/auth/token');

    try {
        const response = await fetch('http://localhost:8000/auth/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams(formData)
        });

        console.log('Status:', response.status);
        const data = await response.json();
        console.log('Response:', data);

        if (response.ok) {
            console.log('✅ Login successful!');
            console.log('Token:', data.access_token);
        } else {
            console.error('❌ Login failed:', data);
        }
    } catch (error) {
        console.error('❌ Network error:', error);
    }
};

testLogin();
