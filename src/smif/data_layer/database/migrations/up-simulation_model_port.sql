CREATE TABLE simulation_model_port(
	"id" serial PRIMARY KEY,
	"model_name" varchar,
	"model_id" integer,
	"port_type" varchar,
	"spec_name" varchar,
	"spec_id" integer
);