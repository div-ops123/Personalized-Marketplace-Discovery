## **What I'd Improve After Deploying the Initial System**

### **1. Version the offline Item Catalog**

The initial implementation assumes item metadata (category, brand, price, tags) changes relatively infrequently, so the training pipeline joins directly to the current item catalog.

After operating the system in production, I'd migrate the offline catalog to a versioned (SCD Type 2) or snapshot-based design. This would enable point-in-time joins for item metadata, preventing historical training examples from using catalog values that changed after the recommendation was served (e.g., price updates or category reclassification).

**Reason:** This improves training-serving consistency for businesses with frequent catalog updates while avoiding unnecessary complexity in the initial implementation.