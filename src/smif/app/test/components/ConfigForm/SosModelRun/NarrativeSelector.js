import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import NarrativeSelector from '../../../../src/components/ConfigForm/SosModelRun/NarrativeSelector.js'

import {sos_model_run, sos_models, narratives, sos_model} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var render, warning

describe('<NarrativeSelector />', () => {

    it('renders narrative sets and narrative name', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={narratives} />)

        expect(render.html()).to.contain(narratives[0].name)
        expect(render.html()).to.contain(narratives[0].narrative_set)
    })

    it('warning no sosModel selected', () => {
        var custom_sos_model_run = Object.assign({}, sos_model_run)
        custom_sos_model_run.sos_model = ''

        render = shallow(<NarrativeSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModel configured in the SosModelRun')

        custom_sos_model_run.sos_model = null

        render = shallow(<NarrativeSelector sosModelRun={custom_sos_model_run} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModel configured in the SosModelRun')
    })

    it('warning no narrativeSets in sosModel', () => {
        var custom_sos_models = sos_models.slice(0)
        custom_sos_models[0].narrative_sets = []

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-info"]')

        expect(warning.html()).to.contain('There are no NarrativeSets configured in the SosModel')

        custom_sos_models[0].narrative_sets = null

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={custom_sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-info"]')

        expect(warning.html()).to.contain('There are no NarrativeSets configured in the SosModel')
    })

    it('warning no sosModelRun configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={empty_object} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')

        render = shallow(<NarrativeSelector sosModelRun={null} sosModels={sos_models} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There is no SosModelRun configured')
    })

    it('warning no sosModels configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={empty_array} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={null} narratives={narratives} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no SosModels configured')
    })

    it('warning no narratives configured', () => {
        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={empty_array} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no Narratives configured')

        render = shallow(<NarrativeSelector sosModelRun={sos_model_run} sosModels={sos_models} narratives={null} />)
        warning = render.find('[id="narrative_selector_alert-danger"]')

        expect(warning.html()).to.contain('There are no Narratives configured')
    })
})
