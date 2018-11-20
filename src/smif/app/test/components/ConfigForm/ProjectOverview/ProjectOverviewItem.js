import React from 'react'
import {expect} from 'chai'
import {shallow, mount} from 'enzyme'
import {describe, it} from 'mocha'
import {MemoryRouter, Route, Switch} from 'react-router-dom'
import sinon from 'sinon'

import ProjectOverviewItem from '../../../../src/components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'
import {empty_array} from '../../../helpers.js'

var item

var itemname = 'item_name'
var items = [
    {
        name: 'item_1',
        description: 'item_description_1'
    },
    {
        name: 'item_2',
        description: 'item_description_2'
    }
]
var itemlink = '/item/link/'

describe.skip('<ProjectOverviewItem />', () => {

    it('renders itemname', () => {
        const wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="row_item_1"]').first()
        expect(item.html()).to.contain('value="' + itemname + '"')
    })

    it('renders items', () => {
        const wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={itemlink} />)

        var item = wrapper.find('[id="row_item_1"]')
        expect(item.html()).to.contain('<td data-name="item_1" class="col-name">' + items[0].name + '</td><td data-name="item_1" class="col-desc">' + items[0].description + '</td>')

        item = wrapper.find('[id="row_item_2"]')
        expect(item.html()).to.contain('<td data-name="item_2" class="col-name">' + items[1].name + '</td><td data-name="item_2" class="col-desc">' + items[1].description + '</td>')
    })

    it('warning no itemname', () => {
        var wrapper = shallow(<ProjectOverviewItem itemname={''} items={items} itemLink={itemlink} />)

        var item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemname configured')

        wrapper = shallow(<ProjectOverviewItem itemname={null} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemname configured')
    })

    it('warning no items', () => {
        var wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={empty_array} itemLink={itemlink} />)

        var item = wrapper.find('[id="project_overview_item_alert-info"]')
        expect(item.html()).to.contain('There are no items in this list')

        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={null} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-info"]')
        expect(item.html()).to.contain('There are no items in this list')
    })

    it('warning no itemLink', () => {
        var wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={''} />)

        var item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemLink configured')

        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={null} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemLink configured')
    })

    it('handles edit on name click', () => {
        const wrapper = mount(<MemoryRouter
            initialIndex={0}
            initialEntries={['/']}>
            <Switch>
                <Route path='/item/link/item_1' render={()=><div id="redirected"></div>} />
                <ProjectOverviewItem
                    itemname={itemname}
                    items={items}
                    itemLink={itemlink} />
            </Switch>
        </MemoryRouter>)
        wrapper.find('td.col-name').first().simulate('click')
        expect(wrapper.find('#redirected')).to.have.length(1)
    })

    it('handles edit on description click', () => {
        const wrapper = mount(<MemoryRouter
            initialIndex={0}
            initialEntries={['/']}>
            <Switch>
                <Route path='/item/link/item_1' render={()=><div id="redirected"></div>} />
                <ProjectOverviewItem
                    itemname={itemname}
                    items={items}
                    itemLink={itemlink} />
            </Switch>
        </MemoryRouter>)
        wrapper.find('td.col-desc').first().simulate('click')
        expect(wrapper.find('#redirected')).to.have.length(1)
    })

    it('calls onDelete from delete button click', () => {
        var onDeleteSpy = sinon.spy()
        const wrapper = mount(<ProjectOverviewItem
            itemname={itemname}
            items={items}
            itemLink={itemlink}
            onDelete={onDeleteSpy} />)
        wrapper.find('button[id="btn_delete_item_2"]').simulate('click')
        expect(onDeleteSpy.calledOnce).to.be.true
        expect(onDeleteSpy.calledWith({
            target: {
                value: 'item_2',
                name: 'item_name',
                type: 'action'
            }
        })).to.be.true
    })

    it('forwards to resultLink on result click', () => {
        const wrapper = mount(<MemoryRouter
            initialIndex={0}
            initialEntries={['/']}>
            <Switch>
                <Route path='/to/results/item_2' render={()=><div id="redirected"></div>} />
                <ProjectOverviewItem
                    itemname={itemname}
                    items={items}
                    itemLink={itemlink}
                    resultLink={'/to/results/'} />
            </Switch>
        </MemoryRouter>)
        wrapper.find('[id="btn_start_item_2"]').first().simulate('click')
        expect(wrapper.find('#redirected')).to.have.length(1)
    })
})
