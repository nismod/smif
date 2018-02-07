import React from 'react'
import sinon from 'sinon'
import { expect } from 'chai'
import { mount, shallow } from 'enzyme'
import TimestepSelector from '../../../../src/components/ConfigForm/SosModelRun/TimestepSelector.js'

import {sos_model_run} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var render, warning, select_resolution, select_baseyear, select_endyear

describe('<TimestepSelector />', () => {

    it('renders resolution', () => {
        render = shallow(<TimestepSelector timeSteps={sos_model_run.timesteps} />)

        const resolution = (sos_model_run.timesteps[sos_model_run.timesteps.length-1] - sos_model_run.timesteps[0]) / (sos_model_run.timesteps.length - 1)
        select_resolution = render.find('[id="sos_model_run_timesteps_resolution"]')

        expect(select_resolution.html()).to.contain('value="' + resolution + '"')
    })

    it('renders baseyear', () => {
        render = shallow(<TimestepSelector timeSteps={sos_model_run.timesteps} />)

        select_baseyear = render.find('[id="sos_model_run_timesteps_baseyear"]')

        expect(select_baseyear.html()).to.contain('value="' + sos_model_run.timesteps[0] + '"')
    })

    it('renders endyear', () => {
        render = shallow(<TimestepSelector timeSteps={sos_model_run.timesteps} />)

        select_endyear = render.find('[id="sos_model_run_timesteps_endyear"]')

        expect(select_endyear.html()).to.contain('value="' + sos_model_run.timesteps[sos_model_run.timesteps.length - 1] + '"')
    })

    it('render without initial timesteps', () => {
        const onChange = sinon.spy()
        render = mount(<TimestepSelector timeSteps={empty_array} onChange={onChange} />)

        select_resolution = render.find('[id="sos_model_run_timesteps_resolution"]')
        select_baseyear = render.find('[id="sos_model_run_timesteps_baseyear"]')
        select_endyear = render.find('[id="sos_model_run_timesteps_endyear"]')

        expect(select_resolution.html()).to.contain('value="5"')
        expect(select_baseyear.html()).to.contain('<option value="2015">')
        expect(select_endyear.html()).to.contain('<option value="2020">')
    })

    it('warning no timeSteps configured', () => {
        const onChange = sinon.spy()
        render = mount(<TimestepSelector timeSteps={null} onChange={onChange} />)
        warning = render.find('[id="timestep_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no TimeSteps configured')
    })
})
