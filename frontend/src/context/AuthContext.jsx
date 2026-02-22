
import { createContext, useContext, useState, useEffect } from 'react';
import client from '../api/client';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(localStorage.getItem('token'));
    const [user, setUser] = useState(null);

    const login = async (username, password) => {
        try {
            // OAuth2 requires application/x-www-form-urlencoded
            // Use URLSearchParams instead of FormData for proper encoding
            const params = new URLSearchParams();
            params.append('username', username);
            params.append('password', password);

            const response = await client.post('/auth/token', params, {
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });

            const { access_token } = response.data;
            setToken(access_token);
            localStorage.setItem('token', access_token);
            setUser({ username }); // In a real app, decode JWT or fetch profile
            return true;
        } catch (error) {
            console.error("Login failed", error);
            throw error;
        }
    };


    const logout = () => {
        setToken(null);
        setUser(null);
        localStorage.removeItem('token');
    };

    return (
        <AuthContext.Provider value={{ token, user, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
