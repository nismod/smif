import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import SosModelRunConfigForm from '../../../src/components/ConfigForm/SosModelRunConfigForm.js'

import {sos_model_run, sos_models, scenarios, narratives} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<SosModelRunConfigForm />', () => {

    const correctRender = shallow(<SosModelRunConfigForm sosModelRun={sos_model_run} sosModels={sos_models} scenarios={scenarios} narratives={narratives} />)
    const dataMissingRender = shallow(<SosModelRunConfigForm sosModelRun={empty_object} sosModels={empty_array} scenarios={empty_array} narratives={empty_array} />)

    it('renders sos_model_run.name', () => {
        const sos_model_run_name = correctRender.find('[id="sos_model_run_name"]')
        expect(sos_model_run_name.html()).to.contain(sos_model_run.name)
    })

    it('renders sos_model_run.name when data missing', () => {
        const sos_model_run_name = dataMissingRender.find('[id="sos_model_run_name"]')
        expect(sos_model_run_name.html()).to.contain(`id="sos_model_run_name"`)
    })

    it('renders sos_model_run.description', () => {
        const sos_model_run_description = correctRender.find('[id="sos_model_run_description"]')
        expect(sos_model_run_description.html()).to.contain(sos_model_run.description)
    })

    it('renders sos_model_run.description when data missing', () => {
        const sos_model_run_description = dataMissingRender.find('[id="sos_model_run_description"]')
        expect(sos_model_run_description.html()).to.contain(`id="sos_model_run_description"`)
    })
})
