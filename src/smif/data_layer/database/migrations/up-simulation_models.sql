CREATE TABLE simulation_models(
	"id" serial PRIMARY KEY,
	"name" varchar,
	"description" varchar,
	"interventions" JSON,
	"wrapper_location" varchar
);