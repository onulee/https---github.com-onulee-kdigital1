import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState } from 'react';

function App() {
  const [count, setCount] = useState(0);
  const [kossie, setKossie] = useState(0);
  useEffect(() => {
    console.log(count)
  }, [count, kossie])

  useEffect(() => {
    console.log('first rendering')
  }, []);
  const increment = () => {
    setCount(count + 1);
  };
  return (
    <div className="App">
      <h1>Kossie Coder</h1>
      <button onClick={increment}>Click</button>
      <button onClick={() => setKossie(kossie + 1)}>Click1</button>
    </div>
  );
}

export default App;
