CREATE TABLE model_runs(
	"id" serial PRIMARY KEY,
	"name" varchar,
	"sos_model" varchar,
	"sos_model_id" integer,
	"scenario_varient" varchar
);