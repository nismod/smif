import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import SosModelSelector from '../../../../src/components/ConfigForm/SosModelRun/SosModelSelector.js'

import {sos_model_run, sos_models} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var render, warning

describe('<SosModelSelector />', () => {

    it('renders all sosModels', () => {
        render = shallow(<SosModelSelector sosModelRun={sos_model_run} sosModels={sos_models} />)

        for (var i in sos_models) {
            expect(render.html()).to.contain('value="' + sos_models[i].name + '">' + sos_models[i].name + '</option>')
        }
    })

    it('selects the sosModel in sosModelRun as default', () => {
        render = shallow(<SosModelSelector sosModelRun={sos_model_run} sosModels={sos_models} />)

        expect(render.html()).to.contain('<option selected="" value="' + sos_model_run.sos_model + '">' + sos_model_run.sos_model + '</option>')
    })

    it('selects no sosModel when none is selected in sosModelRun', () => {
        var custom_sos_model_run = Object.assign({}, sos_model_run)
        custom_sos_model_run.sos_model = ''

        render = shallow(<SosModelSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} />)

        expect(render.html()).to.contain('<option selected="" value="">')
    })

    it('warning no sosModelRun configured', () => {
        render = shallow(<SosModelSelector sosModelRun={empty_object} sosModels={sos_models} />)
        warning = render.find('[id="sos_model_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')

        render = shallow(<SosModelSelector sosModelRun={null} sosModels={sos_models} />)
        warning = render.find('[id="sos_model_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')
    })

    it('warning no sosModelRun configured', () => {
        render = shallow(<SosModelSelector sosModelRun={sos_model_run} sosModels={empty_array} />)
        warning = render.find('[id="sos_model_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')

        render = shallow(<SosModelSelector sosModelRun={sos_model_run} sosModels={null} />)
        warning = render.find('[id="sos_model_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')
    })
})
