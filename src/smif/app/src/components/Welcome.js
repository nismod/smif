import React from 'react'

import IntroBlock from 'components/ConfigForm/General/IntroBlock.js'

const Welcome = () => (
    <IntroBlock title="Welcome to smif" intro="smif (a simulation modelling integration framework) is designed to support the creation and running of system-of-systems models.">
        <div>
            <p>
            The main concern of the framework is to run coupled simulation models of systems. smif takes
            model wrappers configured with inputs and outputs, connects those model wrappers
            into system-of-systems models, and handles running those system-of-systems models with
            various input data and parameter settings.
            </p>
            <p>
            Not sure where to start?
            </p>
            <ul>
                <li>Read the documentation at <a href="http://smif.readthedocs.io/en/latest/?badge=latest">http://smif.readthedocs.io/</a></li>
                <li>Check out the source code or raise an issue on GitHub at <a href="https://github.com/nismod/smif/">https://github.com/nismod/smif/</a></li>
                <li>Contact the developers by raising an issue on GitHub or by <a href="mailto:william.usher@ouce.ox.ac.uk?subject=smif query&amp;cc=tom.russell@ouce.ox.ac.uk">email</a></li>
            </ul>
            <p>
            smif is written and developed at
            the <a href="http://www.eci.ox.ac.uk/" target="_blank" rel="noopener noreferrer">Environmental Change Institute</a>,
            University of Oxford within the EPSRC sponsored MISTRAL programme, as part of
            the <a href="http://www.itrc.org.uk/" target="_blank" rel="noopener noreferrer">Infrastructure Transition Research Consortium.</a>
            </p>
        </div>
    </IntroBlock>
)

export default Welcome
