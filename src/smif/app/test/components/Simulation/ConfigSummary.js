import React from 'react'
import {expect} from 'chai'
import {shallow} from 'enzyme'
import {describe, it} from 'mocha'

import {ModelRunSummary} from '../../../src/components/Simulation/ConfigSummary'
import {sos_model_run} from '../../helpers.js'

describe('<ModelRunSummary />', () => {

    it('renders ModelRunSummary', () => {
        var wrapper = shallow(<ModelRunSummary ModelRun={sos_model_run} />)

        expect(wrapper.html()).to.contain(sos_model_run.decision_module)
        expect(wrapper.html()).to.contain(sos_model_run.name)
        expect(wrapper.html()).to.contain(sos_model_run.narratives['technology'][0])

        expect(wrapper.html()).to.contain(sos_model_run.scenarios['population'])
        expect(wrapper.html()).to.contain(sos_model_run.sos_model)
        expect(wrapper.html()).to.contain(sos_model_run.stamp)
        expect(wrapper.html()).to.contain(sos_model_run.timesteps[0])
    })

})
