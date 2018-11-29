CREATE TABLE simulation_model_port(
	"id" serial PRIMARY KEY,
	"model_name" varchar,
	"model_id" integer,
	"port_type" varchar,
	"specification_name" varchar,
	"specification_id" integer
);