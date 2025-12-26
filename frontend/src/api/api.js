import axios from "axios";

const API = axios.create({
    baseURL: "/api",
    headers: {
        "Content-Type": "application/json",
    },
});

export const getRecommendations = (userId) => API.get(`/recommend/${userId}`);
export const postRating = (payload) => API.post('/rating', payload);
export const getMovies = () => API.get('/movies');
export const getRatedMovies = (userId) => API.get(`/movies/rated/${userId}`);
export const postUser = (payload) => API.post('/users', payload);

export default API;
