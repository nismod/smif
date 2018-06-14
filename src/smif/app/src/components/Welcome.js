import React from 'react';

import IntroBlock from './ConfigForm/General/IntroBlock.js'

const Welcome = () => (
    <IntroBlock title="Welcome to smif" intro="smif (a simulation modelling integration framework) is designed to support the creation and running of system-of-systems models. Aspects of the framework handle inputs and outputs, dependencies between models, persistence of data and the communication of state between timesteps."/>
);

export default Welcome;
