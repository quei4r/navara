# Navara Worker Component

Shared types of the frontend worker pool that crates other than
`navara_worker` read, such as the worker intake limit. Kept apart from
`navara_worker` so that crates `navara_worker` depends on (e.g.
`navara_data_requester`) can use them without a dependency cycle.
