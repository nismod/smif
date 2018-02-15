import React from 'react'
import { expect } from 'chai'
import { shallow, mount } from 'enzyme'
import DependencySelector from '../../../../src/components/ConfigForm/SosModel/DependencySelector.js'

import {sos_model, sector_models} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var wrapper

describe('<DependencySelector />', () => {

    it('warning no sectorModel configured', () => {
        wrapper = shallow(<DependencySelector sectorModels={null} dependencies={sos_model.dependencies} selectedSectorModels={sos_model.sector_models} scenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('sectorModels are undefined')
    })

    it('warning no (selected) dependencies configured', () => {
        wrapper = shallow(<DependencySelector sectorModels={sector_models} dependencies={null} selectedSectorModels={sos_model.sector_models} scenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('Dependencies are undefined')
    })

    it('warning no (selected) sectorModels configured', () => {
        wrapper = shallow(<DependencySelector sectorModels={sector_models} dependencies={sos_model.dependencies} selectedSectorModels={null} scenarioSets={sos_model.scenario_sets}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('selectedSectorModels are undefined')
    })

    it('warning no (selected) scenarioSets configured', () => {
        wrapper = shallow(<DependencySelector sectorModels={sector_models} dependencies={sos_model.dependencies} selectedSectorModels={sos_model.sector_models} scenarioSets={null}/>)
        expect(wrapper.find('[id="dependency_selector_alert-danger"]').html()).to.contain('scenarioSets are undefined')
    })

})
