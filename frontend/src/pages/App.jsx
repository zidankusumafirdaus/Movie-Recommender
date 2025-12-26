

import React, { useState } from 'react';
import '../styles/index.css';
import { getRecommendations } from '../api/api';
import RecommendationList from '../components/RecommendationList';
import Navbar from '../components/Navbar';

function App() {
  const [userId, setUserId] = useState('98');
  const [recommendations, setRecommendations] = useState([]);
  const [strategy, setStrategy] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getRecommendations(userId);
      // backend returns the JSON as described by the user
      const data = res.data;
      setRecommendations(data.recommendations || []);
      setStrategy(data.strategy || '');
    } catch (err) {
      console.error(err);
      setError('Failed to fetch recommendations');
      setRecommendations([]);
      setStrategy('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-primary-900 text-white p-6">
      <Navbar />
      <div className="max-w-1xl mx-auto mt-6">
        <header className="mb-6">
          <h1 className="text-4xl font-extrabold">Movie Recommendations</h1>
          <p className="text-gray-200 mt-2">Personalized movie suggestions powered by collaborative filtering. Explore, rate, and refine your recommendations.</p>
        </header>

        <section className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="md:col-span-2 rounded">
            <div className="flex gap-2 items-center mb-3">
              <input value={userId} onChange={e => setUserId(e.target.value)} className="p-2 rounded bg-white/5 flex-1 focus:outline-none focus:ring-2 focus:ring-primary-500" />
              <button onClick={fetchRecommendations} className="bg-primary-500 hover:bg-primary-700 text-white py-2 px-3 rounded">Get</button>
            </div>

            <div className="mb-3">
              {loading && <p className="text-primary-300">Loading...</p>}
              {error && <p className="text-red-300">{error}</p>}
              {!loading && !error && <p className="text-sm text-primary-300">Strategy: <span className="font-medium">{strategy || '-'}</span></p>}
            </div>

            <RecommendationList recommendations={recommendations} />
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;