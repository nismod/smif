CREATE TABLE sos_model_simulation_models(
	"id" serial PRIMARY KEY,
	"sos_model_name" varchar,
	"sos_model_id" integer,
	"simulation_model_name" varchar,
	"simulation_model_id" integer,
	"sos_sim_model_id" integer
);