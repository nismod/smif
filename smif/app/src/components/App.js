import React from 'react';
import ProjectConfig from './ProjectConfig';
import SectorModelConfig from './SectorModelConfig';
import SosModelConfig from './SosModelConfig';
import SosModelRunConfig from './SosModelRunConfig';

const App = () => (
    <div>
        <h1>Welcome to Smif!</h1>
        <ProjectConfig />
        <SectorModelConfig />
        <SosModelConfig />
        <SosModelRunConfig />
    </div>
);

export default App;
